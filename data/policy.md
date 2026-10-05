# Customer Support Policy — Complaint Handling Rule Book

## 1. Missing Items

- Customers must report missing items **within 24 hours** of delivery.
- Eligible for a refund equal to the missing item's price within the order.
- For combo orders, calculate the **proportional value** of the missing item relative to the combo price.
- Maximum automatic refund for missing items: **$50 per incident**.
- Claims above $50 require human review.

## 2. Wrong Items Delivered

- Customer receives incorrect items due to a packing error.
- Eligible for a **full refund** or a **free redelivery** of the correct items.
- Customer must report within **24 hours** of delivery.
- Photo evidence is required for claims above **$30**.

## 3. Quality Issues

Use the following scale based on reported severity:

| Severity | Description | Refund |
|---|---|---|
| Minor | Slightly cold, minor presentation issue | 20–30% |
| Moderate | Significantly undercooked, clearly poor quality | 50% |
| Severe | Inedible, foreign object, food safety risk | 100% |

- Customers must report quality issues **within 2 hours** of delivery.
- Claims without a description of the quality issue are classified as minor.

## 4. Late Delivery

| Delay | Resolution |
|---|---|
| 30–60 min late | $5 discount voucher |
| 60–90 min late | 20% discount voucher |
| 90+ min late | Full refund |

- Delay is measured from the estimated delivery time shown at checkout.

## 5. Payment & Billing Issues

- Duplicate charges are refunded in full within 2 business days.
- Failed payments with order delivered are treated as billing errors — no refund unless customer provides bank evidence.
- Disputed charges require human review.

## 6. Refund Limits and Caps

- **Maximum automatic refund**: $100 per incident.
- Claims exceeding **$100** require human review regardless of validity.
- Customers with more than **3 refund claims in any 30-day period** require human review for all subsequent claims.

## 7. Fraud Prevention Rules

- Claims where the reported missing item **was not part of the order** are automatically **denied**.
- Repeated claims for the **same issue type within 7 days** require additional photo evidence.
- Risk scores above **50** are escalated to a human reviewer automatically.
- An order that already has `refund` status cannot receive a second refund.

## 8. Voucher Policy

- Vouchers are valid for **30 days** from issue date.
- One voucher per order per incident; vouchers cannot be combined.
- Vouchers are sent to the email address on file.

## 9. Escalation Triggers (Automatic)

Any of the following must trigger an escalation to human review:

1. Claim value > $100
2. Risk score > 50
3. Customer explicitly requests a human agent
4. Order details contradict the complaint (e.g., item not in order)
5. Payment dispute or chargeback mentioned
6. Order has already been refunded
7. Allergen safety flag is active (customer declared allergen AND order contained that allergen)
8. Customer has rejected two or more resolution offers in the same session
9. AI policy retrieval confidence is below 0.60 — do not guess; escalate
10. Customer message is near-identical to previous turn (loop detection)

## 10. Freebie and Promotional Item Policy

- Freebies and promotional add-ons (items given at no charge) are not entitled to a refund.
- If a paid item is missing, the eligible refund is the price of that paid item only.
- Claims where the missing item was a freebie are resolved with an apology and a goodwill coupon (max 5% of order value).
- The distinction between a paid item and a freebie must be verified from order data before any resolution.

## 11. Dietary Mismatch Policy

A dietary mismatch occurs when a customer explicitly states a dietary restriction AND the delivered item violates it.

Conditions required (both must be true):
1. Customer states restriction: vegetarian, vegan, halal, kosher, jain, or similar.
2. At least one delivered item has `veg: false` when the restriction is vegetarian/vegan, or does not meet the stated requirement.

Severity: HIGH. Eligible resolutions: replacement (if in-progress) or partial/full refund.

A customer who simply expresses a preference without tying it to the specific complaint does not trigger this policy.

## 12. Allergen Violation Policy

An allergen violation is a food safety event and is handled differently from other complaints.

Conditions required (both must be true):
1. Customer explicitly declares a specific allergen sensitivity (e.g., "I'm allergic to peanuts").
2. At least one delivered item contains that allergen in its `allergens` list.

Handling:
- Severity: CRITICAL. `allergen_safety = True`.
- Advise the customer NOT to consume the item. State this in the first sentence of the response.
- Always escalate to human review immediately. Do not offer a same-item replacement.
- Full refund is applied automatically alongside escalation.
- This path does NOT route through the standard dietary mismatch branch.

## 13. Repeated Complaints Policy

- If the same customer submits 3 or more complaints in a single session without resolution, escalate.
- If the customer has rejected 2 or more resolution offers in the same session, escalate.
- Repeated identical messages (loop detection) trigger immediate escalation.
- Customers with more than 3 claims in any 30-day period require human review for all subsequent claims in that period.

## 14. Low-Confidence Policy Retrieval

- If the RAG retrieval confidence score is below 0.60, the AI must not invent a policy.
- In this case: issue an empathetic acknowledgment, apply the safest available resolution from the allowed options, and escalate if the safe option does not fully resolve the issue.
- Never present a made-up policy as if it were real.

## 15. Priority Handling (In-Progress Orders)

- Priority handling (expedited re-delivery) may only be offered when the order status is `preparing` or `out_for_delivery`.
- It must NEVER be offered for an order with status `delivered`, `refund`, `partial_refund`, or `cancelled`.
- When offered, the expected wait time is reduced and the customer is not charged an additional delivery fee.
- If the original delay exceeds 30 minutes and the order is still in transit, priority handling is the preferred resolution before offering a refund.

## 16. Agent Behaviour Guidelines

- Always address the customer by name when known.
- Acknowledge the complaint with empathy before requesting details.
- Never promise a refund before completing verification.
- Do not make up order details — if data is unavailable, escalate.
- Provide a clear explanation with every decision (approved, partial, denied, escalated).
