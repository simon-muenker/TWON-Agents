import typing
import pydantic

import cltrier_lib


class PostRequest(pydantic.BaseModel):
    author_first_name: str
    author_last_name: str
    author_party: str
    topic: str

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
            "examples": [
                {
                    "author_first_name": "Steve",
                    "author_last_name": "Scalise",
                    "author_party": "Republican",
                    "topic": "migration, immigrants, borders"
                }
            ]
        }
    )

    def get_instruction(self) -> cltrier_lib.inference.schemas.Chat:
        return cltrier_lib.inference.schemas.Chat(
            messages=[
                cltrier_lib.inference.schemas.Message(
                    role="system",
                    content=f"You are a {self.author_first_name} {self.author_last_name} member of {self.author_party}. Post a Tweet about the following topic:",
                ),
                cltrier_lib.inference.schemas.Message(
                    role="user", content=self.topic
                )
            ]
        ).model_dump()
    

class ReplyRequest(pydantic.BaseModel):
    history: typing.List[cltrier_lib.inference.schemas.Message]
    post: str

    model_config = pydantic.ConfigDict(
        arbitrary_types_allowed=True,
        json_schema_extra={
            "examples": [
                {
                    "history": [
                         {
                            "role": "user",
                            "content": "Our Founding Fathers knew that confidence in the integrity of our elections could best be protected at the local level - not by a federal bureaucrat."
                        },
                        {
                            "role": "assistant",
                            "content": "Republicans believe in local control and common sense. Thats right!"
                        },
                        {
                            "role": "user",
                            "content": "While we acknowledge the magnitude of this ruling, there is still more work to be done. We need to address packer concentration and MCOOL in order to restore transparency and fairness to the cattle market."
                        },
                        {
                            "role": "assistant",
                            "content": "The work is just beginning, and I remain committed to fighting for American ranchers and consumers."
                        },
                    ],
                    "post": "This announcement is a victory for American consumers and producers. The USDAs ruling is a major step in the right direction, and I applaud Secretary of Agriculture Tom Vilsack for taking the necessary actions to fix this label."
                }
            ]
        }
    )

    def get_instruction(self) -> cltrier_lib.inference.schemas.Chat:
        return cltrier_lib.inference.schemas.Chat(
            messages=[
                cltrier_lib.inference.schemas.Message(
                    role="system",
                    content=f"You are a social media user. Respond to the following Tweet based on your last interactions:",
                ),
                *self.history,
                cltrier_lib.inference.schemas.Message(
                    role="user",
                    content=self.post,
                )
            ]
        ).model_dump()