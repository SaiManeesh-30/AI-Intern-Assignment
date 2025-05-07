from transformers import pipeline

class Chatbot:
    def __init__(self):
        # Predefined answers to common customer questions
        self.faq = {
            "latest smartphones specifications": "Our latest smartphones include the X10 with a 6.7-inch AMOLED display, Snapdragon 8 Gen 3 processor, 12GB RAM, and a 50MP camera. The Y20 features a 6.5-inch OLED screen, Dimensity 9300 chip, 8GB RAM, and a 48MP camera.",
            "latest laptops specifications": "The ProBook Z16 has a 16\" OLED display, Intel i9 CPU, 32GB RAM, and 2TB SSD. The UltraNote S14 features a 14\" IPS screen, Ryzen 7 CPU, 16GB RAM, and 1TB SSD.",
            "track my order": "Go to our website's 'Order Tracking' section and enter your order ID and email.",
            "company's return policy": "You can return products within 30 days if they’re in original condition. Full details are on our website.",
            "available payment methods": "We accept Visa, Mastercard, Amex, debit cards, PayPal, and bank transfers.",
            "warranty information": "Most products come with a 1-year warranty. Details are on product pages or in manuals.",
            "accessories we sell": "We stock headphones, chargers, cases, screen protectors, and smartwatches.",
            "shipping costs": "Shipping depends on destination and order weight. Exact fees appear at checkout.",
            "contact customer support": "Reach us at support@examplegadgets.com or 1-800-GADGETS, Mon-Fri, 9 AM - 5 PM EST.",
            "change my order": "Orders are locked after placement. Contact support quickly if changes are needed.",
        }

        self.history = []
        self.classifier = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )

    def respond(self, message):
        self.history.append({"user": message})
        match = self._find_best_match(message)

        if match:
            reply = self.faq[match]
        else:
            sentiment = self.classifier(message)[0]['label']
            if sentiment == "NEGATIVE":
                reply = "It sounds like you're having trouble. Can you clarify your question so I can help?"
            else:
                reply = "Hmm, I'm not sure about that one. Could you ask something else?"

        self.history.append({"bot": reply})
        return reply

    def _find_best_match(self, message):
        # Simple keyword overlap matching
        message_words = set(message.lower().split())
        best_key = None
        best_score = 0

        for question in self.faq:
            key_words = set(question.lower().split())
            overlap = message_words & key_words
            score = len(overlap) / (len(message_words | key_words) + 1e-5)
            if score > best_score:
                best_score = score
                best_key = question

        return best_key if best_score > 0.5 else None

    def clear_history(self):
        self.history = []

    def show_history(self):
        return self.history


def main():
    bot = Chatbot()
    print("Hi there! Ask me anything about our gadgets or services.")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"exit", "quit", "bye"}:
            print("Bot: Thanks for chatting. Take care!")
            break

        reply = bot.respond(user_input)
        print("Bot:", reply)


if __name__ == "__main__":
    main()
