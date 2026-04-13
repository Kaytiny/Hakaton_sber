import os
from gigachat import GigaChat
from gigachat.models import Chat, Messages, MessagesRole


class LLMClient:
    def __init__(self):
        self.credentials = os.environ["GIGACHAT_CREDENTIALS"]
        self.scope = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_CORP")
        self.model = "GigaChat-2-Max"

    def chat(self, system: str, user: str, temperature: float = 0.3) -> str:
        with GigaChat(
            credentials=self.credentials,
            scope=self.scope,
            model=self.model,
            verify_ssl_certs=False,
        ) as giga:
            response = giga.chat(
                Chat(
                    messages=[
                        Messages(role=MessagesRole.SYSTEM, content=system),
                        Messages(role=MessagesRole.USER, content=user),
                    ],
                    max_tokens=2048,
                    temperature=temperature,
                )
            )
        return response.choices[0].message.content.strip()
