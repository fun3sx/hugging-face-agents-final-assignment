import os
from typing import TypedDict, List, Dict, Any, Optional, Annotated
from langgraph.graph import StateGraph, END, MessagesState, START
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from dotenv import load_dotenv
import requests
import json
import re
import time

from tools import *

load_dotenv()

MAX_TOOL_CALLS = 8    # Max total tool calls allowed





class AgentState(TypedDict):
    # Keep message reducer so we can return partial updates
    messages: Annotated[List[Any], add_messages]
    tool_calls: int
    task_id: Optional[str]
    file_name: Optional[str]

def count_tool_calls(messages: List[Any]) -> int:
    """Count the number of tool calls made so far."""
    count = 0
    for message in messages:
        if hasattr(message, 'tool_calls') and message.tool_calls:
            count += len(message.tool_calls)
    return count

def agent():

    tools = [wiki_search, arxiv_search, web_search,
             visit_webpage,wiki_get_section, markdown_table_to_dataframe,
             fetch_task_file, read_text_file,
             transcribe_mp3_openai, load_excel_df, sum_numbers]
    
    
    llm = ChatOpenAI(
        model="gpt-4o",
        temperature=0.1
    )
    '''
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash-lite",
        temperature=0
    )
    '''
    llm_with_tools = llm.bind_tools(tools, parallel_tool_calls=False)
    
    with open("system_prompt.txt", "r", encoding="utf-8") as f:
        system_prompt = f.read().strip()
    
    
    
    def think(state: AgentState) -> AgentState:
        
        context_info = ""
        if state.get("task_id") and state.get("file_name"):
            context_info = f"\n\nCurrent Task Context:\n- Task ID: {state['task_id']}\n- File Name: {state['file_name']}"
        
            sys_msg = SystemMessage(content=system_prompt + context_info)
        else:
            sys_msg = SystemMessage(content=system_prompt)
        
        response = llm_with_tools.invoke([sys_msg] + state["messages"])
        # Update tool call count
        current_tool_calls = count_tool_calls(state["messages"] + [response])
        time.sleep(2)
        
        return {"messages": add_messages(state["messages"], [response]),
                "tool_calls":current_tool_calls}
    
    def should_continue(state: AgentState) -> str:
        """Custom routing function that checks tool call limit."""
        # Count current tool calls
        tool_call_count = count_tool_calls(state["messages"])
        
        # If we've hit the limit, end the conversation
        if tool_call_count >= MAX_TOOL_CALLS:
            return END
        
        # Otherwise, use the standard tools condition
        last_message = state["messages"][-1]
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            return "tools"
        else:
            return END
    
    tools_node = ToolNode(tools)
    
    # Build the graph
    graph = StateGraph(AgentState)
    graph.add_node("think", think)
    graph.add_node("tools", tools_node)
    
    graph.add_edge(START, "think")
    
    # Use conditional edges for routing from think
    graph.add_conditional_edges("think", should_continue)
    
    # After tools execute, return to think for further reasoning
    graph.add_edge("tools", "think")
    
    # Remove the direct edge from think to END - let tools_condition handle it
    
    # Compile
    finished_graph = graph.compile()
    return finished_graph

def extract_final_answer(text):
    # Regular expression to find "FINAL ANSWER:" and capture everything after it
    match = re.search(r"FINAL ANSWER:\s*(.*)", text)
    
    if match:
        return match.group(1).strip()  # Return the captured answer, removing leading/trailing spaces
    else:
        return None  # Return None if "FINAL ANSWER:" isn't found
    
if __name__ == "__main__":
   running_graph = agent() 
   
   DEFAULT_API_URL = "https://agents-course-unit4-scoring.hf.space"

   questions_url = f"{DEFAULT_API_URL}/questions"
   submit_url = f"{DEFAULT_API_URL}/submit"

   QUESTIONS_FILE = "questions.txt"
   
   if os.path.exists(QUESTIONS_FILE):
        with open(QUESTIONS_FILE, "r", encoding="utf-8") as f:
            questions_data = json.load(f)
        print(f"Loaded {len(questions_data)} questions from file.")
   else:
       response = requests.get(questions_url, timeout=15)
       response.raise_for_status()
       questions_data = response.json()
       
       # Save to file
       with open(QUESTIONS_FILE, "w", encoding="utf-8") as f:
            json.dump(questions_data, f, ensure_ascii=False, indent=2)
       print(f"Saved {len(questions_data)} questions to {QUESTIONS_FILE}.")
       

   answers_payload = []
   i=0
   for item in questions_data:
       print ('-'*5)
       print (i)
       print (item)
       
       if i in [17,3,12]:
           i+=1
           continue
       '''
       if i in range(18):
           i+=1
           continue
       
       if i == 19:
           break
       '''
       messages = [HumanMessage(content=item['question'])]
       result = running_graph.invoke({
           "messages": messages , 
           "tool_calls": 0,
           "task_id": item['task_id'],
           "file_name": item['file_name']
           })
       
       '''
       # Print final tool call count
       
       final_count = count_tool_calls(result["messages"])
       print(f"\nTotal tool calls made: {final_count}")
       
       for step in result["messages"]:
           step.pretty_print()
       '''
       
       answers_payload.append({"task_id": item['task_id'],  "submitted_answer": extract_final_answer(result['messages'][-1].content)})
       print (answers_payload[-1])
       
       
           
       i+=1
       time.sleep(60)

   print ('finished with questions')
   
   for item in answers_payload:
       if item['submitted_answer'] is None:
           item['submitted_answer'] = ''

   submission_data = {"username": "fun3s", "agent_code": "https://github.com/fun3sx/hugging-face-agents-final-assignment", "answers": answers_payload}
   
   
   response = requests.post(submit_url, json=submission_data, timeout=60)
   response.raise_for_status()
   result_data = response.json()
   final_status = (
        f"Submission Successful!\n"
        f"User: {result_data.get('username')}\n"
        f"Overall Score: {result_data.get('score', 'N/A')}% "
        f"({result_data.get('correct_count', '?')}/{result_data.get('total_attempted', '?')} correct)\n"
        f"Message: {result_data.get('message', 'No message received.')}"
    )
   
   print(final_status)
        
    
    
    