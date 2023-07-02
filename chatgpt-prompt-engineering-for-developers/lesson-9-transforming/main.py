import os
import openai
# 运行此API配置，需要将目录中的.env中api_key替换为自己的
from dotenv import load_dotenv, find_dotenv
from redlines import  Redlines
_ = load_dotenv(find_dotenv())  # read local .env file
# 导入 OpenAI API_KEY
openai.api_key = os.environ['OPENAI_API_KEY']


def get_completion(prompt, model="gpt-3.5-turbo"):
    messages = [{"role": "user", "content": prompt}]

    response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0,
    )
    return response.choices[0].message["content"]


text = f"""
hello everyone this is Edsion fromA team, 
the idea I bring here today is option trading - new experience  version.
so first question, why we need to build one new user experience?

That is because currently, our option tradeing sysmtem is  strong but complicated.
from ui we can know it has too many fieilds need to fillout. It support more than 10 strategies for user to choose, it is fine for the professional user.
but it is a little difficult for the young trader to learn how to procced.

To simply the trade process, Last year our team buil one app called OSS,
which provides user step by step guidence. 



So long as the customer  fillout less information for the roadmap. 
they will get the contracts they want.

after the user fillout all the information, in first page. second, then click '' in third page. 
Option trade builder will redirect user to 
our option trading system which I introduce in previous slide.


till now, you may have questions come up, why the user need to do this redirect, is is necessary?

Yes, this is where our idea come from. To make the option trading more easily, we want to build a new app which

integrate our custom option trade system with option trade builder to make user

can do the trading in only one app with a lot very friendly guidence.

Other than that, as a new feature, we will provide the web mobile responsive since mobile is popular.

after our app is done. we imagine one scene , new trade lie in bed, use his iphone or pad to bulild contract.

they don't need to understand too many equity knowledge, because the app provide a very helpful guidence already.

after  order is placed, one contract generated in current app, no addition redirect, no addition complex process. all are done.
"""

prompt = f"""
Proofread and correct the following text and rewrite the corrected version
:```{text}```
"""

response = get_completion(prompt)
print(response)

diff = Redlines(text,response)
# display(Markdown(diff.output_markdown))
print(diff.output_markdown)