from typing import Literal

import dspy
import modaic


class EmailIntent(dspy.Signature):
    """
    You are an email classification assistant. Given an email's subject line and body, determine which categories the email belongs to. An email may belong to one or more categories.

    Assign one or more of the following labels:
    - "Business": Work-related correspondence, meetings, proposals, or professional communications
    - "Personal": Messages from friends, family, or personal acquaintances
    - "Promotions": Marketing emails, sales offers, discounts, or advertising
    - "Customer Support": Service inquiries, help requests, ticket updates, or support responses
    - "Job Application": Resumes, cover letters, interview scheduling, or hiring-related correspondence
    - "Finance & Bills": Invoices, payment confirmations, bank statements, or financial notifications
    - "Events & Invitations": Party invites, conference registrations, webinar announcements, or event RSVPs
    - "Travel & Bookings": Flight confirmations, hotel reservations, itineraries, or travel updates
    - "Reminders": Deadline alerts, appointment reminders, follow-ups, or scheduled notifications
    - "Newsletters": Recurring informational digests, blog roundups, or subscription-based content updates

    Return only the labels that apply. If multiple categories fit, include all relevant ones.
    """

    subject: str = dspy.InputField(desc="The subject line of the email")
    body: str = dspy.InputField(desc="The body of the email")
    intent_labels: set[
        Literal[
            "Business",
            "Personal",
            "Promotions",
            "Customer Support",
            "Job Application",
            "Finance & Bills",
            "Events & Invitations",
            "Travel & Bookings",
            "Reminders",
            "Newsletters",
        ]
    ] = dspy.OutputField(desc="The labels that apply to the email")


if __name__ == "__main__":
    arbiter = modaic.Predict(EmailIntent, dspy.LM(model="huggingface/Qwen/Qwen3.5-4B"))
    arbiter.push_to_hub("tyrin/email-intent")
