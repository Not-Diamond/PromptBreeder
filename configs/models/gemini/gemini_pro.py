from opencompass.models.gemini import Gemini


models = [
    dict(abbr='gemini-pro',
         type=Gemini,
         path='gemini-pro',
         key='YOUR_GEMINI_KEY',
         retry=10,
         query_per_second=0.8,
         max_out_len=2048, max_seq_len=2048, batch_size=1
    ),
]
