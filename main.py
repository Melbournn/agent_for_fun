import os
import sys
from langgraph.graph import StateGraph, END
from state import AgentState
from receiver import audio_receiver
from nodes import transcriber, translator, architect, executor

def create_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("transcriber", transcriber)
    workflow.add_node("translator", translator)
    workflow.add_node("architect", architect)
    workflow.add_node("executor", executor)

    workflow.set_entry_point("transcriber")
    workflow.add_edge("transcriber", "translator")
    workflow.add_edge("translator", "architect")
    workflow.add_edge("architect", "executor")
    workflow.add_edge("executor", END)

    return workflow.compile()

def run_agent(audio_path: str):
    if not os.path.exists(audio_path):
        print(f"Error: Audio file not found at {audio_path}")
        return

    print("===== Initializing Workflow =====")
    app = create_graph()
    
    print(f"===== Processing Audio: {audio_path} =====")
    initial_state = audio_receiver(audio_path)
    
    # Run the graph
    final_state = app.invoke(initial_state)
    
    print("\n===== Final Results =====")
    print(f"Transcription: {final_state.get('transcription')}")
    print(f"Translation: {final_state.get('translated_text')}")
    print(f"Image saved to: {final_state.get('image_path')}")
    print(f"Result: {final_state.get('result')}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python main.py <path_to_audio_file>")
    else:
        audio_file = sys.argv[1]
        run_agent(audio_file)
