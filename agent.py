import os
import re
import json

from langchain_core.messages import SystemMessage, AIMessage, HumanMessage, ToolMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
import langchain_google_genai._function_utils as genai_utils

from mcp_client import filter_tool_result


# ── Gemini fix: integer enum values break schema validation ──────────────────
_orig_schema = genai_utils._dict_to_genai_schema

def _patched_schema(schema_dict, *args, **kwargs):
    if isinstance(schema_dict, dict) and "enum" in schema_dict:
        schema_dict["enum"] = [str(x) for x in schema_dict["enum"]]
        if "type" in schema_dict:
            schema_dict["type"] = "string"
    return _orig_schema(schema_dict, *args, **kwargs)

genai_utils._dict_to_genai_schema = _patched_schema


# ── Address comparison (pure Python, no LLM tokens) ─────────────────────────

_PINCODE_RE = re.compile(r'\b([1-9][0-9]{5})\b')


def compare_addresses_by_pincode(swiggy_raw: str, zomato_raw: str) -> dict:
    """Compare delivery locations by extracting 6-digit Indian pincodes."""
    def first_pin(text: str) -> str | None:
        m = _PINCODE_RE.search(str(text))
        return m.group(1) if m else None

    pin_s = first_pin(swiggy_raw)
    pin_z = first_pin(zomato_raw)
    matched = bool(pin_s and pin_z and pin_s == pin_z)
    return {
        "matched": matched,
        "swiggy_pincode": pin_s or "unknown",
        "zomato_pincode": pin_z or "unknown",
        "summary": (
            f"Same delivery area — pincode {pin_s}" if matched
            else f"Different areas — Swiggy: {pin_s or '?'}, Zomato: {pin_z or '?'}"
        ),
    }


# ── Message history repair ────────────────────────────────────────────────────

def _repair_history(messages: list) -> list:
    """
    Enforce Gemini strict turn order: user → assistant → [tools] → user …
    - Inserts placeholder ToolMessages for unanswered tool calls.
    - Drops orphaned ToolMessages with no preceding tool-calling AIMessage.
    """
    out: list = []
    i = 0
    while i < len(messages):
        msg = messages[i]

        if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
            out.append(msg)
            needed_ids = {tc["id"] for tc in msg.tool_calls}
            i += 1
            found_ids: set = set()
            while i < len(messages) and isinstance(messages[i], ToolMessage):
                found_ids.add(messages[i].tool_call_id)
                out.append(messages[i])
                i += 1
            for tid in needed_ids - found_ids:
                print(f"[history] Inserting placeholder for missing tool_call_id={tid}")
                out.append(ToolMessage(content="[Tool response not available]", tool_call_id=tid))

        elif isinstance(msg, ToolMessage):
            prev = out[-1] if out else None
            if isinstance(prev, ToolMessage) or (
                isinstance(prev, AIMessage) and getattr(prev, "tool_calls", None)
            ):
                out.append(msg)
            else:
                print(f"[history] Dropping orphaned ToolMessage id={getattr(msg, 'tool_call_id', '?')}")
            i += 1

        else:
            out.append(msg)
            i += 1

    return out


# ── Tool-result compression ───────────────────────────────────────────────────

# Search-phase tools whose results can be stubbed once ordering starts
_SEARCH_PHASE_TOOLS = frozenset({
    "search_restaurants", "search_menu",
    "get_restaurants_for_keyword", "get_menu_items_listing",
    "get_restaurant_menu", "get_restaurant_menu_by_categories",
    "get_addresses", "get_saved_addresses_for_user",
})

# Any of these in history means selection was made and ordering has begun
_ACTION_PHASE_TOOLS = frozenset({
    "update_food_cart", "create_cart",
    "place_food_order", "checkout_cart",
    "fetch_food_coupons", "apply_food_coupon",
    "get_cart_offers",
})


def _has_id_reference_block(messages: list) -> bool:
    """Return True if any AIMessage already contains a 📋 ID Reference block."""
    for m in messages:
        if isinstance(m, AIMessage) and "📋 ID Reference" in str(m.content):
            return True
    return False


def _compress_old_tool_messages(messages: list) -> list:
    """
    Context compression strategy — three safe tiers:

    TIER 1 — Action phase (cart/checkout/coupon tool seen):
      Keep ALL messages intact.  Every tool result (cart charges, coupon
      offers, checkout confirmation) is still needed for subsequent steps.
      Never compress during the action phase.

    TIER 2 — User has replied AFTER the last search tool result AND
      a 📋 ID Reference block exists in an AIMessage (written by the LLM
      when it presented the comparison table).  The IDs are preserved in
      that block, so the raw search payloads are now safe to stub.

    TIER 3 — Any other case (search in progress, or no 📋 block yet):
      Keep all messages — either the LLM still needs the raw data, or it
      hasn't yet emitted the ID block that makes compression safe.
    """
    # TIER 1 — action phase: never compress anything
    action_seen = any(
        isinstance(m, ToolMessage) and (getattr(m, "name", "") or "") in _ACTION_PHASE_TOOLS
        for m in messages
    )
    if action_seen:
        return messages

    # Find the last ToolMessage index
    last_tool_idx = -1
    for i in range(len(messages) - 1, -1, -1):
        if isinstance(messages[i], ToolMessage):
            last_tool_idx = i
            break

    if last_tool_idx < 0:
        return messages

    # TIER 3 — search still in progress: no human reply after last tool result
    user_replied_after = any(
        isinstance(m, HumanMessage)
        for m in messages[last_tool_idx + 1:]
    )
    if not user_replied_after:
        return messages

    # TIER 3 safety — no 📋 ID Reference block in history yet.
    # The LLM hasn't outputted the IDs, so compressing search results now
    # would leave the LLM with no way to look up the correct IDs.
    if not _has_id_reference_block(messages):
        return messages

    # TIER 2 — user has replied AND 📋 block exists; stub all search-phase results.
    result = []
    for m in messages:
        name = getattr(m, "name", "") or "tool"
        if isinstance(m, ToolMessage) and name in _SEARCH_PHASE_TOOLS:
            result.append(ToolMessage(
                content=f"[{name}: search data compressed — use 📋 ID Reference in prior message]",
                tool_call_id=m.tool_call_id,
                name=name,
            ))
        else:
            result.append(m)
    return result


# ── State modifier (runs before every LLM call) ───────────────────────────────

def _make_prompt(system_text: str, max_history: int = 30):
    """
    Returns a LangGraph state modifier that:
    1. Removes existing SystemMessages.
    2. Applies filter_tool_result to every ToolMessage (truncates lists/chars).
    3. Stubs tool results from older ReAct rounds (_compress_old_tool_messages).
    4. Trims to last max_history messages.
    5. Repairs turn-order violations.
    6. Ensures history starts with a HumanMessage (Gemini requirement).
    7. Guards against empty history.
    8. Prepends the system message.
    """
    sys_msg = SystemMessage(content=system_text)

    def modify(state):
        msgs = state["messages"] if isinstance(state, dict) else list(state)
        msgs = [m for m in msgs if not isinstance(m, SystemMessage)]

        # Build tool_call_id → tool_name from AIMessages.
        # LangGraph sometimes leaves ToolMessage.name empty; the AIMessage
        # tool_calls list is the authoritative source for the actual tool name.
        tc_name_map: dict[str, str] = {}
        for m in msgs:
            if isinstance(m, AIMessage) and getattr(m, "tool_calls", None):
                for tc in m.tool_calls:
                    if tc.get("id") and tc.get("name"):
                        tc_name_map[tc["id"]] = tc["name"]

        # Step 1: filter/truncate raw tool outputs
        filtered = []
        for m in msgs:
            if isinstance(m, ToolMessage):
                # Prefer ToolMessage.name; fall back to the AIMessage tool_calls map
                name = (getattr(m, "name", "") or "") or tc_name_map.get(m.tool_call_id, "")
                # Pass content as-is so filter_tool_result can handle MCP image blocks
                slim = filter_tool_result(name, m.content)
                m = ToolMessage(content=slim, tool_call_id=m.tool_call_id, name=name)
            filtered.append(m)

        # Step 2: stub tool results from previous ReAct rounds
        compressed = _compress_old_tool_messages(filtered)

        trimmed  = compressed[-max_history:]
        repaired = _repair_history(trimmed)

        # Gemini requires history to start with a HumanMessage
        first_human = next((i for i, m in enumerate(repaired) if isinstance(m, HumanMessage)), None)
        if first_human is not None:
            repaired = repaired[first_human:]
        else:
            all_human = [m for m in filtered if isinstance(m, HumanMessage)]
            repaired  = [all_human[-1]] if all_human else []

        if not repaired:
            repaired = [HumanMessage(content="Please continue.")]

        return [sys_msg] + repaired

    return modify


# ── Phase → tool mapping ──────────────────────────────────────────────────────
# Explicit per-platform, per-phase tool names.
# The LLM always has all tools bound (so it can use them), but the system prompt
# tells it exactly which tool to call at each step — preventing cross-phase confusion.

_PHASE_TOOLS = {
    "swiggy": {
        "search":   ["search_restaurants", "search_menu"],
        "cart":     ["update_food_cart"],
        "coupon":   ["fetch_food_coupons", "apply_food_coupon"],
        "checkout": ["place_food_order"],
        "post":     ["get_food_orders", "get_food_order_details", "track_food_order"],
    },
    "zomato": {
        "search":   ["get_restaurants_for_keyword", "get_menu_items_listing"],
        "cart":     ["create_cart"],
        "coupon":   ["get_cart_offers"],
        "checkout": ["checkout_cart"],
        "post":     ["get_order_history", "get_order_tracking_info"],
    },
}


def _tool_names(platform: str, phase: str) -> str:
    return ", ".join(_PHASE_TOOLS.get(platform, {}).get(phase, ["—"]))


# ── System prompt ─────────────────────────────────────────────────────────────

def _build_system_prompt(
    has_swiggy: bool,
    has_zomato: bool,
    address_context: dict | None,
    address_ids: dict | None,
    addr_comparison: dict | None,
    payment_types: dict | None = None,
) -> str:
    both            = has_swiggy and has_zomato
    address_context = address_context or {}
    address_ids     = address_ids     or {}

    # ── Platform scope ────────────────────────────────────────────────────────
    if both:
        platform_scope = "Both Swiggy and Zomato are connected. Search BOTH for every food query."
    elif has_swiggy:
        platform_scope = "Only Swiggy is connected. Use Swiggy tools only."
    else:
        platform_scope = "Only Zomato is connected. Use Zomato tools only."

    # ── Mandatory parameters ──────────────────────────────────────────────────
    # Inject literal, pre-verified address IDs so the LLM copies them verbatim.
    # Do NOT show raw JSON — show only the exact value to use.
    param_lines = ["You MUST pass these exact values in EVERY tool call that accepts an address:"]
    if has_swiggy:
        s_id    = address_ids.get("swiggy", "")
        s_label = ""
        try:
            s_label = json.loads(address_context.get("swiggy", "{}")).get("label", "")
        except Exception:
            pass
        if s_id:
            line = f'  Swiggy  →  parameter name: addressId    value: "{s_id}"'
            if s_label:
                line += f"   [{s_label}]"
            param_lines.append(line)
        else:
            param_lines.append("  Swiggy  →  addressId not resolved — call get_addresses first")

    if has_zomato:
        z_id    = address_ids.get("zomato", "")
        z_label = ""
        try:
            z_label = json.loads(address_context.get("zomato", "{}")).get("label", "")
        except Exception:
            pass
        if z_id:
            line = f'  Zomato  →  parameter name: address_id   value: "{z_id}"'
            if z_label:
                line += f"   [{z_label}]"
            param_lines.append(line)
        else:
            param_lines.append("  Zomato  →  address_id not resolved — call get_saved_addresses_for_user first")

    param_lines.append("")
    param_lines.append("Payment method:")
    if has_swiggy:
        param_lines.append('  Swiggy  →  paymentMethod = "Cash"    (used in place_food_order — fixed, COD)')
    if has_zomato:
        z_pay_opts = (payment_types or {}).get("zomato", [])
        opts_str = ", ".join(f'"{v}"' for v in z_pay_opts) if z_pay_opts else '"upi_qr", "pay_later"'
        param_lines.append(f'  Zomato  →  payment_type  = ask the user at Step 3. Options from MCP: {opts_str}')
    param_lines.append("")
    param_lines.append("Do NOT modify, omit, or guess any of these values. Copy them verbatim at every step.")
    if both and addr_comparison:
        param_lines.append(f"  Address match: {addr_comparison['summary']}")
    param_block = "\n".join(param_lines)

    # ── Payment options (read live from MCP schema) ───────────────────────────
    z_pay_opts = (payment_types or {}).get("zomato", [])
    if z_pay_opts:
        z_pay_opts_str = "  " + "\n  ".join(
            f"{i+1}. {v}" for i, v in enumerate(z_pay_opts)
        )
    else:
        z_pay_opts_str = "  1. upi_qr\n  2. pay_later"

    # ── Phase-by-phase tool guide ─────────────────────────────────────────────
    # Tell the LLM exactly which tool to call at each step. This prevents it from
    # calling checkout tools during search, or passing cart IDs to search tools.
    if both:
        tool_guide = f"""TOOL GUIDE — follow this exact sequence:

STEP 1  SEARCH ROUND 1 — call ALL THREE tools in a SINGLE turn (parallel):
  • search_restaurants(addressId="<from above>", query="<user query>")
  • search_menu(addressId="<from above>", query="<user query>")
  • get_restaurants_for_keyword(address_id="<from above>", keyword="<user query>")

  *** You MUST call all three in one turn. Do NOT call just one and stop. ***
  *** Getting search_restaurants results is NOT enough — Zomato search is also MANDATORY. ***

STEP 1b  SEARCH ROUND 2 — after Round 1 completes, call:
  • get_menu_items_listing(address_id="<from above>", res_id=<restaurant id from get_restaurants_for_keyword>)

  If Swiggy returns SESSION_EXPIRED_403: skip Swiggy tools, continue Zomato only.
  If Zomato returns SESSION_EXPIRED_403: skip Zomato tools, continue Swiggy only.

STEP 2  PRESENT RESULTS
  Show up to 10 rows per platform, each row = one food item from a DIFFERENT restaurant.
  Sort each table by Price ASCENDING (cheapest first). Use IDENTICAL column format for both tables.
  Every cell MUST be filled — use "—" only when data is truly unavailable.

  Column data sources:
  SWIGGY  — each row comes from search_menu results:
    #          = row number (1–10)
    Restaurant = restaurant name from search_menu (match restaurant_id → search_restaurants)
    Area       = area/locality from search_restaurants
    Item       = item name from search_menu
    Price      = item price from search_menu (e.g. ₹199)
    Rating     = restaurant rating from search_restaurants
    Delivery   = delivery time from search_restaurants

  ZOMATO  — each row comes from get_menu_items_listing results:
    #          = row number (1–10)
    Restaurant = restaurant name from get_restaurants_for_keyword
    Area       = area/locality from get_restaurants_for_keyword
    Item       = item name from get_menu_items_listing
    Price      = item price from get_menu_items_listing
    Rating     = restaurant rating from get_restaurants_for_keyword
    Delivery   = delivery time from get_restaurants_for_keyword

  ## 🟠 Swiggy Options
  | # | Restaurant | Area | Item | Price | Rating | Delivery Time |
  |---|-----------|------|------|-------|--------|---------------|
  | 1 | ...       | ...  | ...  | ₹...  | ⭐...  | ~... min      |

  ## 🔴 Zomato Options
  | # | Restaurant | Area | Item | Price | Rating | Delivery Time |
  |---|-----------|------|------|-------|--------|---------------|
  | 1 | ...       | ...  | ...  | ₹...  | ⭐...  | ~... min      |

  After both tables, add a short **💡 Recommendation** section:
  - Pick the best value option (balance of price, rating, delivery time).
  - If Sangam Restaurant appears in either list for chicken biryani, highlight it as the preferred choice.
  - Explain the recommendation in 1–2 sentences.

  Ask: "Reply with **Swiggy <number>** or **Zomato <number>** to place the order (e.g. Swiggy 3)."

  MANDATORY — immediately after the tables and recommendation, output this ID REFERENCE BLOCK
  (it is the ONLY thing that survives search compression, so it MUST be complete and accurate):

  📋 ID Reference (preserved for ordering — do not delete):
  Swiggy: 1. [item name] | restaurant_id=[X] | menu_item_id=[Y] | price=₹[Z] | restaurant=[name]
          2. [item name] | restaurant_id=[A] | menu_item_id=[B] | price=₹[C] | restaurant=[name]
          ... (one line per row shown in the Swiggy table)
  Zomato: 1. [item name] | res_id=[P] | variant_id=[Q] | price=₹[R] | restaurant=[name]
          2. [item name] | res_id=[S] | variant_id=[T] | price=₹[U] | restaurant=[name]
          ... (one line per row shown in the Zomato table)

  Fill in REAL values from the tool results — never use placeholders like [X] or [Y].
  This block MUST list every item in the tables so when the user picks "Swiggy 3",
  the correct restaurant_id/menu_item_id can be read from line 3 even after search data is removed.

  IMPORTANT — after the user picks an item, ALL search/restaurant results from this step
  are no longer needed. They will be removed from context automatically to save tokens.

STEP 3  CONFIRM ORDER
  Show item, restaurant, price, delivery address. Ask user to confirm.
  Then present payment options based on the platform chosen:

  If user chose Swiggy:
    Payment options:
      1. Cash on Delivery
    Ask: "Choose your payment method (reply with the number)." Wait for reply.
    (Always use paymentMethod="Cash" in place_food_order regardless of which number they pick.)

  If user chose Zomato:
    Payment options from MCP:
{z_pay_opts_str}
    Ask: "Choose your payment method (reply with the number)." Wait for reply.
    Store their choice as payment_type for Step 4.

STEP 4  BUILD CART
  Swiggy : {_tool_names("swiggy", "cart")}
    restaurantId*   : string  — restaurant_id from search_restaurants result
    cartItems*      : array   — [{{"menu_item_id": "<id from search_menu>", "quantity": 1, "variants": []}}]
    addressId*      : use the exact value from MANDATORY PARAMETERS above
    restaurantName  : string  — restaurant name (optional, pass if available)

  Zomato : {_tool_names("zomato", "cart")}
    res_id*         : integer — restaurant id from get_restaurants_for_keyword result
    items*          : array   — [{{"variant_id": "<id starting with v_ from menu result>", "quantity": 1, "add_ons": []}}]
    address_id*     : use the exact value from MANDATORY PARAMETERS above
    payment_type*   : use the value chosen by the user in Step 3 ("upi_qr" or "pay_later")
    promo_code      : omit unless a coupon was applied

  → Save the cart_id / cartId from the response — required for Steps 5 and 6.

  After the cart tool returns, extract and display the bill breakdown from the response:

  🧾 **Order Bill:**
  | | |
  |---|---|
  | Item Total | ₹[item_total] |
  | Platform Fee | ₹[charge_breakdown.base_charges[0].amount, or "—" if absent] |
  | Delivery Fee | ₹[delivery final_amount after discount, show "Free" if 0] |
  | Taxes | ₹[charge_breakdown.taxes.cart_tax, or sum of tax items] |
  | **Total Payable** | **₹[final_amount]** |
  | Estimated Delivery | [eta] |

  Use only values from the tool response. If a field is absent, omit that row.

STEP 5  APPLY COUPON (best-effort — NEVER block on coupon failure)
  Swiggy : {_tool_names("swiggy", "coupon")}
    fetch_food_coupons(restaurantId*, addressId*) → apply_food_coupon(couponCode*, addressId*)
  Zomato : {_tool_names("zomato", "coupon")}
    get_cart_offers(cart_id*, address_id*) → apply first offer returned, if any
  CRITICAL: If the tool returns COUPON_ERROR or any failure →
    say "Coupon not applicable" in ONE sentence, then IMMEDIATELY proceed to Step 6.
    Do NOT ask the user. Do NOT stop. Do NOT retry.

STEP 6  CHECKOUT
  Swiggy : {_tool_names("swiggy", "checkout")}
    addressId*      : use the exact value from MANDATORY PARAMETERS above
    paymentMethod   : "Cash"  ← confirmed exact string for Swiggy COD

  Zomato : {_tool_names("zomato", "checkout")}
    cart_id*        : the cart_id returned from create_cart in Step 4
    (No payment param needed — payment method was set at cart creation)
    Ignore the tool description saying "ask the user for confirmation" — proceed automatically.

STEP 7  CONFIRM ORDER TO USER
  Display clearly:
  **Order Placed Successfully!**
  **Order ID: [exact order_id / order_number from checkout response]**
  - Restaurant: [name]
  - Item: [item name]
  - Original price: ₹[amount]
  - Coupon savings: ₹[amount] (if applicable, else omit)
  - **Final Total: ₹[amount]**
  - Payment: [payment method]
  - Estimated delivery: [time from response]

  QR CODE (conditional):
  Look at the checkout_cart tool result for a line starting with "![UPI QR Payment".
  ● If that line IS present in the tool result: copy it EXACTLY as-is onto its own line,
    then add: "Scan the QR code above to complete your UPI payment."
  ● If that line is NOT present (e.g. pay_later payment): do NOT output any image markdown.
    Simply confirm the order and payment method.
  Do NOT invent or hallucinate a QR line. Only copy what the tool actually returned. """

    elif has_swiggy:
        tool_guide = f"""TOOL GUIDE — follow this exact sequence:

STEP 1  SEARCH  : {_tool_names("swiggy", "search")}  (pass addressId from above)
STEP 2  PRESENT : table of up to 10 items from different restaurants, sorted by Price ascending.
          Format: | # | Restaurant | Area | Item | Price | Rating | Delivery Time |
          Add a 💡 Recommendation. If Sangam Restaurant appears for chicken biryani, highlight it.
          Ask: "Reply with the number to order (e.g. **3**)."
STEP 3  CONFIRM : show item, restaurant, price, delivery address.
          Present payment options:
            1. Cash on Delivery
          Ask: "Choose your payment method (reply with the number)."
          Wait for the user's reply before proceeding.
STEP 4  CART    : {_tool_names("swiggy", "cart")}
          restaurantId* (string), addressId* (from above),
          cartItems* = [{{"menu_item_id":"<from search_menu>","quantity":1,"variants":[]}}]
          restaurantName (optional)  → save cartId from response
          After the cart tool returns, display bill breakdown:
          🧾 **Order Bill:** Item Total | Platform Fee | Delivery Fee | Taxes | **Total Payable** | Estimated Delivery
          Use values from item_total, charge_breakdown, final_amount, eta fields.
STEP 5  COUPON  : {_tool_names("swiggy", "coupon")}
          If COUPON_ERROR → "Coupon not applicable", immediately go to Step 6. No retry.
STEP 6  CHECKOUT: {_tool_names("swiggy", "checkout")}
          addressId* (from above), paymentMethod="Cash"
STEP 7  CONFIRM : **Order ID: [id]**, restaurant, item, totals, coupon savings, Cash on Delivery, delivery time"""

    else:
        tool_guide = f"""TOOL GUIDE — follow this exact sequence:

STEP 1  SEARCH  : {_tool_names("zomato", "search")}  (pass address_id from above)
STEP 2  PRESENT : table of up to 10 items from different restaurants, sorted by Price ascending.
          Format: | # | Restaurant | Area | Item | Price | Rating | Delivery Time |
          Add a 💡 Recommendation. If Sangam Restaurant appears for chicken biryani, highlight it.
          Ask: "Reply with the number to order (e.g. **3**)."
STEP 3  CONFIRM : show item details — ask user to confirm.
          Then present payment options from MCP:
{z_pay_opts_str}
          Ask: "Choose your payment method (reply with the number)." Wait for reply before proceeding.
STEP 4  CART    : {_tool_names("zomato", "cart")}
          res_id* (integer), address_id* (from above), payment_type*=<user's choice from Step 3>,
          items* = [{{"variant_id":"<v_... from menu>","quantity":1,"add_ons":[]}}]
          → save cart_id from response
          After the cart tool returns, display bill breakdown:
          🧾 **Order Bill:** Item Total | Platform Fee | Delivery Fee | Taxes | **Total Payable** | Estimated Delivery
          Use values from item_total, charge_breakdown, final_amount, eta fields.
STEP 5  COUPON  : {_tool_names("zomato", "coupon")}
          If COUPON_ERROR → "Coupon not applicable", immediately go to Step 6. No retry.
STEP 6  CHECKOUT: {_tool_names("zomato", "checkout")}
          cart_id* (from Step 4 — no payment param needed, was set in cart)
STEP 7  CONFIRM : **Order ID: [id]**, restaurant, item, totals, coupon savings, payment method, delivery time.
  QR CODE (conditional):
  Look at the checkout_cart tool result for a line starting with "![UPI QR Payment".
  ● If that line IS present: copy it EXACTLY as-is onto its own line,
    then add: "Scan the QR code above to complete your UPI payment."
  ● If NOT present: do NOT output any image markdown. Confirm order and payment method only.
  Do NOT invent or hallucinate a QR line. Only copy what the tool actually returned."""

    return f"""You are FoodCompare AI — a food search and ordering assistant for Swiggy and Zomato.

{platform_scope}

USER PREFERENCE: The user prefers ordering Chicken Biryani from Sangam Restaurant.
When Sangam Restaurant appears in search results for biryani, always highlight it in the
recommendation and suggest it as the preferred choice.

━━━ MANDATORY PARAMETERS ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{param_block}
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{tool_guide}

SELECTION CONTINUITY — critical for turn-to-turn correctness:
When the user replies with a selection (e.g. "Swiggy 3" or "Zomato 5"):
  1. Look up row 3/5 in the 📋 ID Reference block you output at STEP 2, OR in the raw search
     tool results if they are still present in context.
  2. Before calling any cart tool, echo back in plain text:
       "Selected: [item name] from [restaurant name] on [Platform]
        restaurant_id/res_id = <value>  |  menu_item_id/variant_id = <value>  |  price = ₹<value>"
  3. Then immediately proceed to STEP 3 (confirm + payment method).
  This echo ensures the key IDs are captured in the conversation history even
  after the large search payloads are removed from context.
  FALLBACK: If you cannot find the IDs (no 📋 block and no search results), call the
  search tools again (STEP 1) with the same query before proceeding — never guess IDs.

RULES:
- If a tool returns SESSION_EXPIRED_403(swiggy): immediately stop all Swiggy tool calls,
  switch to Zomato-only mode, and complete the full flow (search → cart → coupon → checkout)
  using Zomato tools. Inform the user: "Swiggy session expired — placing order on Zomato instead."
- If a tool returns SESSION_EXPIRED_403(zomato): immediately stop all Zomato tool calls,
  switch to Swiggy-only mode, and complete the full flow using Swiggy tools.
  Inform the user: "Zomato session expired — placing order on Swiggy instead."
- If BOTH platforms return SESSION_EXPIRED_403: say "Both sessions have expired — please re-login via the sidebar."
- If any tool returns 500: say "Platform temporarily unavailable" and try the other platform.
- Do NOT retry a failed tool more than once.
- Never invent prices, restaurant names, or item details — only use what tools return.
- The addressId / address_id values above are pre-verified. Never substitute another value.
- Always pass the address parameter at every step: search, cart, coupon, AND checkout.
"""


# ── Agent factory ─────────────────────────────────────────────────────────────

def create_food_aggregator_agent(
    mcp_tools: list,
    connected_platforms: list = None,
    address_context: dict = None,
    address_ids: dict = None,
    address_comparison: dict = None,
    payment_types: dict = None,
):
    connected_platforms = connected_platforms or []
    has_swiggy = "swiggy" in connected_platforms
    has_zomato = "zomato" in connected_platforms

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite"),
        google_api_key=os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"),
        temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.1")),
        max_tokens=int(os.getenv("GEMINI_MAX_TOKENS", "4096")),
    )

    system_text = _build_system_prompt(
        has_swiggy, has_zomato, address_context, address_ids, address_comparison,
        payment_types=payment_types,
    )

    return create_react_agent(
        model=llm,
        tools=mcp_tools,
        prompt=_make_prompt(system_text),
        checkpointer=MemorySaver(),
    )
