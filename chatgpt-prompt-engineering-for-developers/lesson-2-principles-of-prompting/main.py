import os
import openai
# 运行此API配置，需要将目录中的.env中api_key替换为自己的
from dotenv import load_dotenv, find_dotenv

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


# principle 1 write clear and specific instructions

# 1.use prompt 可以避免prompt injection
# 2.use delimiters
# 3.ask for the structured output
# 4.check whether conditions are satisfied
# 5.few-shot prompting - Give successful examples of completing tasks
# Then ask model to perform the task

text = f"""
You should express what you want a model to do by \
providing instructions that are as clear and \
specific as you can possibly make them. \
This will guide the model towards the  desired output, \
and reduce the changes of receiving irrelevant \
or incorrect responses. Don't confuse writing a \
clear prompt with writing a short prompt. \
In many cases, longer prompts provide more clarity \
and context for the model, which can lead to \
more detailed and relevant outputs.
"""

prompt = f"""
Summarize the text delimited by triple backticks \
into a single sentence.
```{text}```
"""

# response = get_completion(prompt)
# print(response)


prompt = f"""
Generate a list of US supermarket name, along with their creator, and number of employees, \
The list should contains 3 elements.
Provide them  in JSON format with the following keys:
supermarket_name,  creator, num_of_emp
"""
# response = get_completion(prompt)
# print(response)

text = f"""
鱼香肉丝的做法：
主料：精瘦猪肉200克，笋丝50克，木耳50克，青椒丝50克，胡萝卜50克。
辅料：料酒5克，盐3克，姜丝适量，水淀粉3克，蒜末适量，蚝油5克，生抽5克，白糖5克，食用油5毫升，葱花适量，郫县豆豉10克。
葱姜、料酒、水淀粉腌制猪瘦肉。调鱼香酱汁。锅内热油下葱爆香，放入腌好的肉炒至变色。加入鱼香汁，配菜大火翻炒。装盘出锅，好吃的鱼香肉丝就做好了。
"""

prompt = f"""
You will be provided with text delimited by triple backticks. \
If it contains a sequence of instructions, \
re-write those instructions in the following format:

Step 1 - ..
Step 2 - ...
...
Step N - ...
If the text doesn't contain a sequence of instructions, \
then simply write \"No steps provided.\"

```{text}```
"""

# response = get_completion(prompt)
# print(response)

prompt = f"""
You task is to answer in a consistent style.

<child>:Teach me  about patience.
<grandparent>: 一位禅师在走夜路的时候碰到了一位提着灯笼的盲人，路过的人都说：“这个盲人真奇怪，明明看不见，却每天晚上提着灯笼。”
禅师对此感到很奇怪，便去问这位盲人。
盲人告诉他，他什么也看不见，但他从来没有被别人碰到过。因为他的灯笼既为别人照亮了前方的路，也让别人看到了他，这样，别的路人也就不会因为看不见而撞到他了。
耐心，是根植于内心的修养，修养是根植于骨子里的高贵，高贵是为别人着想的善良。
凡事多点耐心，多为别人着想，最后收获的不仅是别人，还有我们自己。
有一回，我和家人去超市采买日常用品。在结账时，收银台前已排起了长队，我们只好站在一旁等候结账。
尽管周末买东西的人很多，但超市的收银员却是忙而不乱，耐心地在帮每一位顾客结账，使得队伍仍在有序地向前移动着。
可轮到一位中年妇人结账时，队伍却停滞了。因为这位中年妇人不是用的现钞，而是用一大摞的一角钱硬币。
由于需要将它们一个一个地推叠成一元的硬币，结算起来颇费工夫。
起初，人们依旧有序地等待，但渐渐地，队伍里开始有了不耐烦的情绪，并有了抱怨声。
“没关系，钱当然要数清楚，您慢慢来，大家会理解的 。”说话的是收银的小姑娘，她的话很是贴心，既安抚了妇人，又体现了对其他人的尊重。
过了一会，姑娘又道：“要我帮您吗？”中年妇人点点头。得到了许可，小姑娘便熟练地帮着妇人堆叠着。
后来，在小姑娘的帮助下，这位中年妇人终于将钱币堆叠好，并结算了购买的账单。整个过程中，这位小姑娘的脸上没有出现过一丝不耐烦的表情。

<child>:Teach me about resilience.
"""
# response = get_completion(prompt)
# print(response)

# principle 2 Give the model time to think
# 1.specify the steps to complete a task
#     step-1
#     step-2
#     step-3
#  2.Instruct the model to work out its own solution before rushing to a conclusion


text = f"""
In a charming village, siblings Jack and Jill set out on \
a quest to fetch water from a hilltop \
well, As they climbed, singing joyfully, misfortune \
struck-Jack tripped on a stone and tumbled \
down the hill , with Jill following suit. \
Though slightly battered, the pair returned home to \
comforting embraces. Despite the mishap, \
their adventurous spirits remained undimmed, and  they \
continued exploring with delight.  
"""

prompt_1 = f"""
Perform the following actions:
1 - Summarize the following text delimited by tripple \
backticks with 1 sentence.
2 - Translate the summary into French.
3 - List each name in the French summary.
4 - Output a json object that contains the following \
keys: french_summary, num_names.

Separated your answers with line breaks.

Text:
```{text}```
"""
response = get_completion(prompt_1)
print("Completion for prompt 1:")
print(response)


# Model Limitations
# Hallucination
# Makes statements that sound plausible
# but are not true
#
# Reducing hallucinations:
# First find relevant information
# then answer the question
# based on the relevant information
