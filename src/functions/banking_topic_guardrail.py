import mlrun
import mlrun.serving
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Any, Optional


class LLMModelServer(mlrun.serving.LLModel):
    def __init__(
        self,
        context: mlrun.MLClientCtx = None,
        name: str = None,
        model_path: str = None,
        model_name: str = None,
        **kwargs
    ):
        super().__init__(name=name, context=context, model_path=model_path, **kwargs)
        self.model_name = model_name
    
    def load(
        self,
    ):
        # Load the model from Hugging Face
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_name)


    def predict(
        self,
        body: Any,
        messages: dict = None,
        invocation_config: Optional[dict] = None,
        **kwargs,
    ) -> Any:
        print("This is the body i got:", body)
        print("This is the messages i got:", messages)

        messages_str = " ".join([message["content"] for message in messages])
        print("This is the messages_str i filtered:", messages_str)

        input_ids, attention_mask = self.tokenizer(
            messages_str, return_tensors="pt"
        ).values()

        outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, max_new_tokens=5)
        # Remove input:
        outputs = self.tokenizer.decode(outputs[0])
        print("This is the outputs i filtered:", outputs)
        decoded_outputs_split = outputs.split(body["question"])[-1]
        if "True" in decoded_outputs_split:
            answer = "True"
        elif "False" in decoded_outputs_split:
            answer = "False"
        else:
            answer = "Unavailable"
        return {"question": body["question"], "answer": answer}
