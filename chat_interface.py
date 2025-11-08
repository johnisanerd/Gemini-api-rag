"""
Interactive Chat Interface for YouTube Channel RAG
"""

import sys
from typing import Optional
from gemini_manager import GeminiManager


class ChatInterface:
    def __init__(self, gemini_manager: GeminiManager):
        """
        Initialize chat interface

        Args:
            gemini_manager: GeminiManager instance with loaded transcripts
        """
        self.gemini_manager = gemini_manager

    def start_chat(self):
        """
        Start the interactive chat session
        """
        print("\n" + "="*70)
        print("🎙️  YouTube Channel RAG Chat Interface")
        print("="*70)
        print("\nYou can now ask questions about the video transcripts!")
        print("Type 'exit', 'quit', or 'q' to end the chat.")
        print("Type 'help' for available commands.")
        print("-"*70 + "\n")

        while True:
            try:
                # Get user input
                user_input = input("You: ").strip()

                # Check for exit commands
                if user_input.lower() in ['exit', 'quit', 'q']:
                    print("\n👋 Thanks for chatting! Goodbye!")
                    break

                # Check for help command
                if user_input.lower() == 'help':
                    self._show_help()
                    continue

                # Check for info command
                if user_input.lower() == 'info':
                    self._show_store_info()
                    continue

                # Skip empty input
                if not user_input:
                    continue

                # Get AI response
                print("\n🤖 Assistant: ", end="", flush=True)
                response = self.gemini_manager.chat_with_transcripts(user_input)
                print(response)
                print("\n" + "-"*70 + "\n")

            except KeyboardInterrupt:
                print("\n\n👋 Chat interrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\n❌ Error: {str(e)}")
                print("Please try again or type 'exit' to quit.\n")

    def _show_help(self):
        """Show help information"""
        print("\n" + "="*70)
        print("Available Commands:")
        print("="*70)
        print("  help  - Show this help message")
        print("  info  - Show information about the current file search store")
        print("  exit  - Exit the chat (also: quit, q)")
        print("\nTips:")
        print("  • Ask specific questions about the video content")
        print("  • Request summaries of topics across multiple videos")
        print("  • Ask for comparisons between different videos")
        print("  • Query for specific timestamps or quotes")
        print("="*70 + "\n")

    def _show_store_info(self):
        """Show file search store information"""
        info = self.gemini_manager.get_store_info()
        print("\n" + "="*70)
        print("File Search Store Information:")
        print("="*70)
        for key, value in info.items():
            print(f"  {key}: {value}")
        print("="*70 + "\n")


def run_chat(gemini_manager: GeminiManager):
    """
    Run the chat interface

    Args:
        gemini_manager: GeminiManager instance with loaded transcripts
    """
    chat = ChatInterface(gemini_manager)
    chat.start_chat()
