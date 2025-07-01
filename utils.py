import re
import os
import time
import imageio

from decimal import Decimal
from collections import deque
import matplotlib.pyplot as plt
import traceback

import numpy as np
import gymnasium as gym
from openai import OpenAI
from jinja2 import Template
import google.generativeai as genai
import anthropic


class EpisodeRewardBufferNoBias:
    def __init__(self, max_size):
        self.buffer = deque(maxlen=max_size)

    def add(self, weights: np.ndarray, reward):
        self.buffer.append((weights, reward))

    def sort(self):
        self.buffer = deque(
            sorted(self.buffer, key=lambda x: x[1], reverse=False),
            maxlen=self.buffer.maxlen,
        )

    def __str__(self):
        buffer_table = "Parameters | Reward\n"
        for weights, reward in self.buffer:
            buffer_table += f"{weights.reshape(1, -1)} | {reward}\n"
        return buffer_table

    def load(self, folder):
        # Find all episode files
        all_files = [
            os.path.join(folder, x)
            for x in os.listdir(folder)
            if x.startswith("warmup_rollout")
        ]
        all_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))

        # Load parameters from all episodes
        for filename in all_files:
            with open(filename, "r") as f:
                lines = f.readlines()
                parameters = []
                for line in lines:
                    if "parameter ends" in line:
                        break
                    try:
                        parameters.append([float(x) for x in line.split(",")])
                    except:
                        continue
                parameters = np.array(parameters)

                rewards = []
                for line in lines:
                    if "Total reward" in line:
                        try:
                            rewards.append(float(line.split()[-1]))
                        except:
                            continue
                rewards_mean = np.mean(rewards)
                self.add(parameters, rewards_mean)
                f.close()
        print(self)


class LinearPolicyNoBias:
    def __init__(self, dim_states, dim_actions):
        self.dim_states = dim_states
        self.dim_actions = dim_actions

        self.weight = np.random.rand(self.dim_states, self.dim_actions)

    def initialize_policy(self):
        self.weight = np.round(
            (np.random.rand(self.dim_states, self.dim_actions) - 0.5) * 6, 1
        )

    def get_action(self, state):
        state = state.T
        return np.matmul(state, self.weight)

    def __str__(self):
        output = "Weights:\n"
        for w in self.weight:
            output += ", ".join([str(i) for i in w])
            output += "\n"

        return output

    def update_policy(self, weight_and_bias_list):
        if weight_and_bias_list is None:
            return
        self.weight = np.array(weight_and_bias_list)
        self.weight = self.weight.reshape(-1)
        for i in range(len(self.weight)):
            self.weight[i] = Decimal(self.weight[i]).normalize()

        self.weight = self.weight.reshape(self.dim_states, self.dim_actions)

    def get_parameters(self):
        return self.weight


class LLMBrain:
    def __init__(
        self,
        llm_si_template: Template,
        llm_output_conversion_template: Template,
        llm_model_name: str,
    ):
        self.llm_si_template = llm_si_template
        self.llm_output_conversion_template = llm_output_conversion_template
        self.llm_conversation = []
        assert llm_model_name in [
            "o1-preview",
            "gpt-4o",
            "gemini-2.0-flash-exp",
            "gpt-4o-mini",
            "gemini-1.5-flash",
            "gemini-1.5-flash-8b",
            "gemini-1.5-pro",
            "gemini-2.5-pro-preview-05-06",
            "gemini-2.5-flash-preview-04-17",
            "o3-mini-2025-01-31",
            "gpt-4o-2024-11-20",
            "gpt-4o-2024-08-06",
            "claude-3-7-sonnet-20250219",
        ]
        self.llm_model_name = llm_model_name
        if "gemini" in llm_model_name:
            self.model_group = "gemini"
            genai.configure(api_key=os.environ["GEMINI_API_KEY"])
        elif "claude" in llm_model_name:
            self.model_group = "anthropic"
            self.client = anthropic.Client(api_key=os.environ["ANTHROPIC_API_KEY"])
        else:
            self.model_group = "openai"
            self.client = OpenAI()

    def reset_llm_conversation(self):
        self.llm_conversation = []

    def add_llm_conversation(self, text, role):
        if self.model_group == "openai":
            self.llm_conversation.append({"role": role, "content": text})
        elif self.model_group == "anthropic":
            self.llm_conversation.append({"role": role, "content": text})
        else:
            self.llm_conversation.append({"role": role, "parts": text})

    def query_llm(self):
        for attempt in range(10):
            try:
                if self.model_group == "openai":
                    completion = self.client.chat.completions.create(
                        model=self.llm_model_name,
                        messages=self.llm_conversation,
                    )
                    response = completion.choices[0].message.content
                elif self.model_group == "anthropic":
                    message = self.client.messages.create(
                        model=self.llm_model_name,
                        messages=self.llm_conversation,
                        max_tokens=1024,
                    )
                    response = message.content[0].text
                else:
                    model = genai.GenerativeModel(model_name=self.llm_model_name)
                    chat_session = model.start_chat(history=self.llm_conversation[:-1])
                    response = chat_session.send_message(
                        self.llm_conversation[-1]["parts"]
                    )
                    response = response.text
            except Exception as e:
                print(f"Error: {e}")
                print("Retrying...")
                if attempt == 9:
                    raise Exception("Failed")
                else:
                    print("Waiting for 60 seconds before retrying...")
                    time.sleep(60)

            if self.model_group == "openai":
                # add the response to self.llm_conversation
                self.add_llm_conversation(response, "assistant")
            else:
                self.add_llm_conversation(response, "model")

            return response

    def parse_parameters(self, parameters_string):
        new_parameters_list = []

        # Update the Q-table based on the new Q-table
        for row in parameters_string.split("\n"):
            if row.strip().strip(","):
                try:
                    parameters_row = [
                        float(x.strip().strip(",")) for x in row.split(",")
                    ]
                    new_parameters_list.append(parameters_row)
                except Exception as e:
                    print(e)

        return new_parameters_list

    def llm_update_parameters_num_optim(
        self,
        episode_reward_buffer,
        parse_parameters,
        step_number,
        rank=None,
        optimum=None,
        search_step_size=0.1,
        actions=None,
    ):
        self.reset_llm_conversation()

        system_prompt = self.llm_si_template.render(
            {
                "episode_reward_buffer_string": str(episode_reward_buffer),
                "step_number": str(step_number),
                "rank": rank,
                "optimum": str(optimum),
                "step_size": str(search_step_size),
                "actions": actions,
            }
        )

        self.add_llm_conversation(system_prompt, "user")

        api_start_time = time.time()
        new_parameters_with_reasoning = self.query_llm()
        api_time = time.time() - api_start_time
        new_parameters_list = parse_parameters(new_parameters_with_reasoning)

        return (
            new_parameters_list,
            "system:\n"
            + system_prompt
            + "\n\n\nLLM:\n"
            + new_parameters_with_reasoning,
            api_time,
        )


def find_file(dir):
    return os.path.join(dir, "overall_log.txt")


def read_file(file):
    with open(file, "r") as f:
        lines = f.readlines()

    cpu_times = []
    api_times = []
    all_total_episodes = []
    all_total_steps = []
    total_rewards = []
    for line in lines:
        iteration, cpu_time, api_time, total_episodes, total_steps, total_reward = (
            line.strip().split(",")
        )
        try:
            iteration = int(iteration)
        except:
            continue

        cpu_times.append(eval(cpu_time))
        api_times.append(eval(api_time))
        all_total_episodes.append(eval(total_episodes))
        all_total_steps.append(eval(total_steps))
        total_rewards.append(eval(total_reward))
    return cpu_times, api_times, all_total_episodes, all_total_steps, total_rewards


def plot_data(total_episodes, total_rewards, title):
    plt.plot(np.array(total_episodes), total_rewards, label=title)
    plt.xlabel("Epoch")
    plt.ylabel("Total Rewards")
    plt.title(title)
    if "pong" in title.lower():
        plt.ylim(-1, 4)
    elif "nav" in title.lower():
        plt.ylim(-500, 4500)
    elif "cartpole" in title.lower():
        plt.ylim(-100, 600)
    elif "mountaincar" in title.lower():
        plt.ylim(-500, 200)
    elif "invertedpendulum" in title.lower():
        plt.ylim(-100, 1200)
    # plt.legend()
    plt.show()
    # plt.savefig("tmp.jpeg")
