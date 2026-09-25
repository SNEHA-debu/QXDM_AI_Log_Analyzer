
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


MODEL_NAME = "Qwen/Qwen2.5-1.5B-Instruct"


# --------------------------------------------------
# Load Qwen GenAI model
# --------------------------------------------------

qwen_tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME
)

qwen_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)


# --------------------------------------------------
# Generate grounded technical explanation
# --------------------------------------------------

def generate_ai_explanation(analysis):

    system_message = """
You are a telecom engineering assistant analyzing synthetic
QXDM-style LTE/5G modem log results.

STRICT RULES:

1. Use ONLY facts and measurements supplied by the analysis.
2. Never invent possible causes that are not present in the data.
3. Never claim that the ML prediction is a confirmed root cause.
4. Mention all significant supplied measurements.
5. This is synthetic QXDM-style data.
6. Keep the response concise.
7. Do not generate names of protocol messages that are not
   explicitly supplied in the input.
8. Do not speculate about RF interference, SIM problems,
   network configuration, UE capability, security configuration,
   hardware problems, or coverage.
"""

    user_message = f"""
Generate a technical analysis using ONLY these facts:

UE ID: {analysis['ue_id']}
Total Events: {analysis['total_events']}
Failure Events: {analysis['total_failures']}
Failure Rate: {analysis['failure_rate'] * 100:.1f}%
Authentication Failures: {analysis['authentication_failures']}
Handover Failures: {analysis['handover_failures']}
Total Retransmissions: {analysis['total_retransmissions']}
Average Retransmission: {analysis['avg_retransmission']}
ML Predicted Category: {analysis['predicted_root_cause']}

Write exactly these three sections:

Observed Behavior:
Mention all supplied measurements.

ML Prediction:
State exactly:
"Predicted Category: {analysis['predicted_root_cause']}"

Then state exactly:
"This is a model prediction, not a confirmed network root cause."

Recommended Investigation:

Write exactly these three bullet points:

- Review authentication-related signaling associated with the
  observed authentication failures.

- Review handover-related signaling associated with the
  observed handover failure.

- Review retransmission behavior associated with the observed
  retransmission measurements.

Do not add possible causes, explanations, examples, protocol
messages, or additional recommendations.
"""

    messages = [
        {
            "role": "system",
            "content": system_message
        },
        {
            "role": "user",
            "content": user_message
        }
    ]

    text = qwen_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    model_inputs = qwen_tokenizer(
        [text],
        return_tensors="pt"
    ).to(qwen_model.device)

    generated_ids = qwen_model.generate(
        **model_inputs,
        max_new_tokens=300,
        do_sample=False
    )

    generated_ids = [
        output_ids[len(input_ids):]
        for input_ids, output_ids
        in zip(
            model_inputs.input_ids,
            generated_ids
        )
    ]

    explanation = qwen_tokenizer.batch_decode(
        generated_ids,
        skip_special_tokens=True
    )[0]

    return explanation
