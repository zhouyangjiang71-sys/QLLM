import json
import os
import re
from openai import OpenAI
import torch
import torch.nn.functional as F
import torch.nn as nn
import traceback
import numpy as np
class QLLM_maker():
    def __init__(self,args,system_prompt):
        self.args = args
        self.messages = [{"role": "system", "content": system_prompt}]
        self.client=OpenAI(
            api_key="<your api key>",
            base_url="https://api.deepseek.com"
        )
    def remove_prefix(self,input_str):
        match = re.search(r'```python(.*?)```', input_str, re.DOTALL)
        if match:
            return match.group(1).strip()
        return None
    def maker(self,prompt):
        if self.messages[-1]["role"] == "user":
            self.messages[-1] = {"role": "user", "content": prompt}
        else:
            self.messages.append({"role": "user", "content": prompt})

        while True:
            try:
                response = self.client.chat.completions.create(
                    model="<model name>",
                    messages=self.messages,
                )
                break
            except Exception:
                print(traceback.format_exc())
                pass
        for tim in range(11):
            if tim==11:
                os._exit(os.EX_OK)
            try:
                content = self.remove_prefix(response.choices[0].message.content)
                if content is None:
                    raise ValueError("Your output has to contain code that starts with ```python and ends with ```")
                local_namespace = {}
                exec(content, globals(), local_namespace)
                QLLMNetwork = local_namespace['QLLMNetwork']
                x=QLLMNetwork(torch.zeros(60, self.args.n_agents).cuda(), torch.zeros(60, self.args.state_shape).cuda())
                if x.shape != torch.Size([60, 1]):
                    raise ValueError(f"Output dimension error: expected torch.Size([batchsize, 1]), but got {x.shape}.")
                if torch.isnan(x).any():
                    raise ValueError(f"Invalid numbers are appearing in the output global Q-value, you have an abnormal number of weights, so please double-check that you don't have a problem with dividing by zero!")
                break
            except Exception:
                print(f"Function execution error !!!!!!!!!Function execution error !!!!!!!!Function execution error !!!!!!!{traceback.format_exc()}")
                print(response.choices[0].message.content)
                while True:
                    try:
                        response = self.client.chat.completions.create(
                            model="<model name>",
                            messages=self.messages,
                        )
                        break
                    except Exception:
                        print(traceback.format_exc())

                        pass
        print(content)
        return content
    def addmemory(self,memory):
        self.messages.append({"role": "assistant", "content": memory})

class LLM_evaluator():
    def __init__(self,setup_prompt,map):
        self.map = map
        self.system_prompt = """
        You are an excellent evaluator of multi-agent reinforcement learning performance. I will attempt different functions in multi-agent reinforcement learning to solve the credit assignment problem.
        This function is required to compute the global Q-values accurately without training. It must not use trainable layers such as nn.Linear.
        Your task is to propose modifications to the function in brief plain text suggestions(A little piece will do) or return function selection results in an array format. 
        Your suggestions must be in line with the task description and requirements, and should not be based on random assumptions or unrealistic scenarios.
        Please listen carefully to my subsequent instructions and do not provide additional answers!
        When you output plain text suggestions, please ensure that you do not output any selections. When you output selections in an array format, please ensure that you do not output plain text suggestions.
        Whether you output plain text suggestions or selections, consider if the function meets all the details and considerations specified in the task description. 
        Do not use any information that has not been provided to you to answer! The multi-agent reinforcement learning task description is as follows:  
        """+setup_prompt
        self.messages = [{"role": "system", "content": self.system_prompt}]
        self.client = OpenAI(
            api_key="<your api key>",
            base_url="https://api.deepseek.com"
        )
    def maker(self,prompt):
        self.messages.append({"role": "user", "content": prompt})
        while True:
            try:
                response = self.client.chat.completions.create(
                    model="<model name>",
                    messages=self.messages
                )
                break
            except Exception:
                print(traceback.format_exc())
                pass
        self.messages.append({"role": "assistant", "content": response.choices[0].message.content})
        print(response.choices[0].message.content)
        return response.choices[0].message.content

class LLM_args_manage:
    def __init__(self):
        self.map = "3m"
        self.n_agents=3
        self.state_shape=90
        self.maker_num=3

def smooth(data, sm):
    z = np.ones(len(data))
    y = np.ones(sm) * 1.0
    smooth_data = np.convolve(y, data, "same") / np.convolve(y, z, "same")
    return list(smooth_data)
def loadQLLM(QLLMnetwork):
    with open('src/QLLM_frame.txt', 'r', encoding='utf-8') as file:
        frame = file.read()
    lines = QLLMnetwork.splitlines()
    def_line = 0
    for j in range(len(lines)):
        if 'def' in lines[j]:
            def_line=j
            break
    for i in reversed(range(len(lines))):
        if 'return' in lines[i]:
            lines[i] = lines[i].replace('return ', 'global_q=')
            break
    lines=lines[(def_line+1):]
    indented_lines = ['    ' + line for line in lines]
    QLLMnetwork = "\n".join(indented_lines)
    with open('src/modules/mixers/qllm.py', 'w') as file:
        file.write(frame+"\n"+QLLMnetwork+"\n    "+"    return (global_q*agents_q.shape[-1]).reshape(a, b, 1).cuda()")
