from opencompass.models import OpenAI


api_meta_template = dict(
    round=[
        dict(role='HUMAN', api_role='HUMAN'),
        dict(role='BOT', api_role='BOT', generate=True),
    ],
)

models = [
    dict(abbr='gpt-4',
         type=OpenAI,
         path='gpt-4',
         key="ENV",  # The key will be obtained from $OPENAI_API_KEY, but you can write down your key here as well
         meta_template=api_meta_template,
         query_per_second=1,
         retry=10,
         max_out_len=2048, max_seq_len=4096, batch_size=1),
]
