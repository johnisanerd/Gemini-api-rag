#!/usr/bin/env python3
"""
YouTube Channel RAG Tool
Main application to scrape YouTube channels and chat with video transcripts using Gemini API
"""

import os
import sys
from dotenv import load_dotenv
from youtube_scraper import YouTubeScraper
from gemini_manager import GeminiManager
from chat_interface import run_chat


def load_environment():
    """
    Load environment variables from .env file

    Returns:
        Tuple of (gemini_api_key, apify_api_token)
    """
    load_dotenv()

    gemini_api_key = os.getenv("GEMINI_API_KEY")
    apify_api_token = os.getenv("APIFY_API_TOKEN")

    if not gemini_api_key:
        print("❌ Error: GEMINI_API_KEY not found in .env file")
        print("Please create a .env file with your Gemini API key")
        print("Get your API key from: https://aistudio.google.com/app/apikey")
        sys.exit(1)

    if not apify_api_token:
        print("❌ Error: APIFY_API_TOKEN not found in .env file")
        print("Please add your Apify API token to the .env file")
        print("Get your token from: https://console.apify.com/account/integrations")
        sys.exit(1)

    return gemini_api_key, apify_api_token


def get_user_input():
    """
    Get channel URL and video count from user

    Returns:
        Tuple of (channel_url, max_videos)
    """
    print("\n" + "="*70)
    print("🎬 YouTube Channel RAG Tool")
    print("="*70)
    print("\nThis tool will:")
    print("  1. Scrape videos from a YouTube channel")
    print("  2. Extract video transcripts/subtitles")
    print("  3. Upload them to Gemini API for semantic search")
    print("  4. Let you chat with the video content using AI")
    print("\n" + "-"*70)

    # Get channel URL
    while True:
        print("\n📺 Enter the YouTube channel URL:")
        print("   Examples:")
        print("   - https://www.youtube.com/@channelname")
        print("   - https://www.youtube.com/c/channelname")
        print("   - https://www.youtube.com/channel/UCxxxxx")
        channel_url = input("\nChannel URL: ").strip()

        if channel_url:
            break
        print("⚠️  Please enter a valid URL")

    # Get number of videos
    while True:
        print("\n📊 How many videos to process? (from newest to oldest)")
        print("   Recommended: 5-20 videos for faster processing")
        try:
            max_videos_input = input("\nNumber of videos: ").strip()
            max_videos = int(max_videos_input)

            if max_videos <= 0:
                print("⚠️  Please enter a positive number")
                continue

            if max_videos > 50:
                confirm = input(f"\n⚠️  Processing {max_videos} videos may take a while. Continue? (y/n): ")
                if confirm.lower() != 'y':
                    continue

            break
        except ValueError:
            print("⚠️  Please enter a valid number")

    return channel_url, max_videos


def main():
    """
    Main application flow
    """
    try:
        # Load environment variables
        gemini_api_key, apify_api_token = load_environment()

        # Get user input
        channel_url, max_videos = get_user_input()

        # Initialize scrapers
        print("\n" + "="*70)
        print("🚀 Starting the process...")
        print("="*70)

        youtube_scraper = YouTubeScraper(apify_api_token)
        gemini_manager = GeminiManager(gemini_api_key)

        # Step 1: Scrape YouTube channel
        videos = youtube_scraper.scrape_channel(channel_url, max_videos)

        if not videos:
            print("\n❌ No videos found or scraped. Please check the channel URL.")
            sys.exit(1)

        # Format video data
        formatted_videos = youtube_scraper.format_video_data(videos)

        # Step 2: Create file search store
        store_name = f"YouTube_Channel_{channel_url.split('/')[-1][:30]}"
        gemini_manager.create_file_search_store(store_name)

        # Step 3: Upload transcripts
        uploaded_files = gemini_manager.upload_video_transcripts(formatted_videos)

        if not uploaded_files:
            print("\n❌ No transcripts were uploaded. Videos may not have subtitles.")
            sys.exit(1)

        # Step 4: Start chat interface
        print("\n" + "="*70)
        print("✅ Setup Complete!")
        print("="*70)
        print(f"\n📚 Processed {len(uploaded_files)} video transcripts")
        print("🎉 Ready to chat!\n")

        run_chat(gemini_manager)

    except KeyboardInterrupt:
        print("\n\n👋 Process interrupted. Goodbye!")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
