import pandas as pd
import random
from pathlib import Path

random.seed(42)

CATEGORIES = {
    "billing": [
        "I was charged twice for my subscription",
        "My invoice amount is incorrect",
        "Why was my card debited again?",
        "Billing error on my account",
        "My payment failed but money was deducted",
        "I need my receipt for last month",
        "Your charges are too high, explain the bill",
    ],
    "delivery": [
        "My order has not arrived yet",
        "The delivery is late",
        "Package tracking is not updating",
        "My parcel was delivered to the wrong address",
        "The rider did not call me",
        "Delivery took too long",
        "Where is my package?",
    ],
    "refund": [
        "I want a refund for my purchase",
        "How do I get my money back?",
        "Refund is taking too long",
        "My refund request was rejected",
        "I returned the product but no refund yet",
        "I was promised a refund but nothing happened",
        "Refund my payment immediately",
    ],
    "technical_support": [
        "The app keeps crashing when I open it",
        "I cannot log into the system",
        "The website is not loading properly",
        "My account dashboard is blank",
        "The app is slow and freezing",
        "I get an error message when I try to pay",
        "The system keeps timing out",
    ],
    "account": [
        "I forgot my password and cannot reset it",
        "Please change my email address",
        "My account was locked unexpectedly",
        "I want to deactivate my account",
        "Someone accessed my account without permission",
        "I cannot verify my account",
        "My profile details are not saving",
    ],
    "general_inquiry": [
        "What are your working hours?",
        "Do you offer discounts?",
        "How can I contact customer support?",
        "Where is your office located?",
        "What services do you provide?",
        "How do I upgrade my subscription?",
        "Tell me more about your pricing plans",
    ]
}

def generate_ticket(category: str) -> str:
    base_text = random.choice(CATEGORIES[category])

    noise = [
        "please help",
        "urgent",
        "asap",
        "this is frustrating",
        "I need assistance",
        "kindly respond",
        "thanks",
        "I'm disappointed",
        "this is unacceptable",
        "help me resolve this"
    ]

    if random.random() < 0.7:
        base_text = base_text + " " + random.choice(noise)

    if random.random() < 0.4:
        base_text = base_text + " " + random.choice(noise)

    return base_text.lower()


def create_dataset(n_samples: int = 500) -> pd.DataFrame:
    rows = []
    categories = list(CATEGORIES.keys())

    for _ in range(n_samples):
        category = random.choice(categories)
        text = generate_ticket(category)
        rows.append({"text": text, "label": category})

    df = pd.DataFrame(rows)
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def main():
    project_root = Path(__file__).resolve().parents[2]
    output_path = project_root / "data" / "raw" / "support_tickets.csv"

    df = create_dataset(n_samples=700)
    df.to_csv(output_path, index=False)

    print("Dataset created successfully!")
    print(f"Saved to: {output_path}")
    print(df.head(10))


if __name__ == "__main__":
    main()
