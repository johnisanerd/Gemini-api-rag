"""
Gemini API File Search Manager
Handles file uploads and vector store creation for RAG
"""

import os
import tempfile
from typing import List, Dict
from google import genai
from google.genai import types


class GeminiManager:
    def __init__(self, api_key: str):
        """
        Initialize Gemini manager with API key

        Args:
            api_key: Google Gemini API key
        """
        self.client = genai.Client(api_key=api_key)
        self.file_search_store = None

    def create_file_search_store(self, store_name: str) -> str:
        """
        Create a new file search store

        Args:
            store_name: Display name for the store

        Returns:
            Store name/ID
        """
        print(f"\n📚 Creating file search store: {store_name}")

        self.file_search_store = self.client.file_search_stores.create(
            config={'display_name': store_name}
        )

        print(f"✅ File search store created: {self.file_search_store.name}")
        return self.file_search_store.name

    def upload_video_transcripts(self, videos: List[Dict]) -> List[str]:
        """
        Upload video transcripts to the file search store

        Args:
            videos: List of video data with transcripts

        Returns:
            List of uploaded file names
        """
        if not self.file_search_store:
            raise ValueError("File search store not created. Call create_file_search_store() first.")

        print(f"\n📤 Uploading {len(videos)} video transcripts...")

        uploaded_files = []

        for idx, video in enumerate(videos, 1):
            title = video.get('title', 'Unknown')
            transcript = video.get('transcript', '')
            video_url = video.get('url', '')
            description = video.get('description', '')

            if not transcript:
                print(f"⚠️  Skipping '{title}' - No transcript available")
                continue

            # Create a formatted transcript file
            content = self._format_transcript(video)

            # Create a temporary file
            with tempfile.NamedTemporaryFile(
                mode='w',
                suffix='.txt',
                delete=False,
                encoding='utf-8'
            ) as temp_file:
                temp_file.write(content)
                temp_file_path = temp_file.name

            try:
                # Upload to file search store
                print(f"  [{idx}/{len(videos)}] Uploading: {title[:50]}...")

                operation = self.client.file_search_stores.upload_to_file_search_store(
                    file=temp_file_path,
                    file_search_store_name=self.file_search_store.name,
                    config={
                        'display_name': f"{title[:100]}.txt",
                        'chunking_config': {
                            'white_space_config': {
                                'max_tokens_per_chunk': 500,
                                'max_overlap_tokens': 50
                            }
                        }
                    }
                )

                # Wait for upload to complete
                while not operation.done():
                    pass

                uploaded_files.append(title)
                print(f"  ✅ Uploaded successfully")

            finally:
                # Clean up temp file
                os.unlink(temp_file_path)

        print(f"\n✅ Successfully uploaded {len(uploaded_files)} transcripts!")
        return uploaded_files

    def _format_transcript(self, video: Dict) -> str:
        """
        Format video data into a readable transcript file

        Args:
            video: Video data dictionary

        Returns:
            Formatted transcript text
        """
        title = video.get('title', 'Unknown Title')
        url = video.get('url', '')
        description = video.get('description', '')
        transcript = video.get('transcript', '')
        published_date = video.get('published_date', '')
        view_count = video.get('view_count', 0)

        content = f"""# {title}

**Video URL:** {url}
**Published:** {published_date}
**Views:** {view_count:,}

## Description
{description}

## Transcript
{transcript}
"""
        return content

    def chat_with_transcripts(self, user_question: str) -> str:
        """
        Chat with the video transcripts using Gemini

        Args:
            user_question: User's question

        Returns:
            AI response
        """
        if not self.file_search_store:
            raise ValueError("File search store not created. Upload transcripts first.")

        response = self.client.models.generate_content(
            model="gemini-2.0-flash-exp",
            contents=user_question,
            config=types.GenerateContentConfig(
                tools=[types.Tool(
                    file_search=types.FileSearch(
                        file_search_store_names=[self.file_search_store.name]
                    )
                )],
                temperature=0.7,
            )
        )

        return response.text

    def get_store_info(self) -> Dict:
        """
        Get information about the current file search store

        Returns:
            Store information
        """
        if not self.file_search_store:
            return {"status": "No store created"}

        return {
            "name": self.file_search_store.name,
            "display_name": self.file_search_store.display_name,
        }
