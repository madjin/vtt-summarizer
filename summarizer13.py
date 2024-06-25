import os
import glob
import argparse
import csv
import webvtt
import re
from datetime import datetime, timedelta
from collections import defaultdict
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOllama

# Initialize the LLM model
local_llm = 'phi3:14b-medium-4k-instruct-q5_K_M'
print(f"Initializing LLM for extraction: {local_llm}")
model = ChatOllama(model=local_llm, temperature=0.1)

# Define prompt templates for each pass
overview_prompt = """
You carefully provide accurate, factual, thoughtful, nuanced responses, and are brilliant at reasoning. Only output the summary which can include questions asked, any interesting quotes, and any action items that were discussed. 

Analyze the following meeting transcript. Provide a comprehensive summary paragraph or two (250-400 words) that captures the key points discussed, decisions made, and the overall purpose of the meeting:

Transcript:
{transcript}

Format your response as a single or double paragraph without bullet points. Do not repeat the prompt back, or say anything extra.
"""

main_topics_prompt = """
List 5-10 main topics discussed in the following meeting transcript. Include a timestamp for each topic:

Transcript:
{transcript}

Format your response as:

- Topic 1 (HH:MM:SS)
- Topic 2 (HH:MM:SS)
...

Only output the bullet points per section which can include questions asked, any interesting quotes, and any action items that were discussed.
Do not repeat the prompt back, or say anything extra.
Do not include any preamble, introduction or postscript about what you are doing. Assume I know.
"""

decisions_impact_prompt = """
Extract 25-50 decisions from the following meeting transcript, including their potential impact and timestamps:

Transcript:
{transcript}

Format your response as:

- **Decision**: [Decision 1]
  - **Impact**: [Detailed impact of Decision 1]
- **Decision**: [Decision 2]
  - **Impact**: [Detailed impact of Decision 2]
...

Only output the bullet points per section which can include questions asked, any interesting quotes, and any action items that were discussed. Do not repeat the prompt back, or say anything extra.

Do not include any preamble, introduction or postscript about what you are doing. Assume I know.
"""

action_items_prompt = """
List 5-10 action items from the following meeting transcript, including who they're assigned to (if mentioned), a detailed description, and timestamp:

Transcript:
{transcript}

Format your response as:

- **Item**: [Action item 1]
  - **Assigned to**: [Person assigned]
  - **Description**: [Detailed description of the action item]
...

Only output the bullet points per section which can include questions asked, any interesting quotes, and any action items that were discussed. Do not repeat the prompt back, or say anything extra.

Do not include any preamble, introduction or postscript about what you are doing. Assume I know.
"""

post_processing_prompt = """
Review the following meeting summary sections and reformat them according to these strict guidelines:
1. Provide exactly four main sections: Meeting Overview, Main Topics, Decisions and Impact, and Action Items.
2. Do not add any additional headers or subheaders within these sections.
3. Keep the Meeting Overview as a single paragraph without bullet points.
4. Use bullet points (-) for all items within the Main Topics, Decisions and Impact, and Action Items sections.
5. Ensure all timestamps are in the format (HH:MM:SS).
6. Remove any unnecessary brackets or parentheses from the Main Topics.
7. Combine similar points and remove redundancies.
8. Keep the content single-spaced.
9. Do not repeat the prompt back, or say anything extra.
10. Do not include any preamble, introduction or postscript about what you are doing.

Original content:
{content}

Without repeating the prompt, or including a preamble, simply reformat the content to match this exact structure:

Meeting Overview
[Single or double paragraph summary]

Main Topics
- Topic 1 (HH:MM:SS)
- Topic 2 (HH:MM:SS)
...

Decisions and Impact
- **Decision**: Decision 1
  - **Impact**: Impact of Decision 1
- **Decision**: Decision 2
  - **Impact**: Impact of Decision 2
...

Action Items
- **Item**: Action item 1
  - **Assigned to**: Person assigned
  - **Description**: Description of action item
...

"""

def read_csv_file(file_path):
    print(f"Reading CSV data from file: {file_path}")
    participants = []
    with open(file_path, mode='r', encoding='utf-8-sig') as file:
        csv_reader = csv.reader(file)
        for _ in range(4):  # Skip the first 4 lines
            next(csv_reader)
        for row in csv_reader:
            if len(row) >= 3:
                participants.append({"name": row[0].strip(), "total_duration": row[2].strip()})
    return participants

def read_vtt_file(file_path):
    print(f"Reading transcript from file: {file_path}")
    vtt = webvtt.read(file_path)
    transcript = " ".join([caption.text for caption in vtt])
    print(f"Transcript length: {len(transcript)}")
    return transcript

def format_timestamp(timestamp):
    try:
        if 't=' in timestamp:
            seconds = int(timestamp.split('=')[1].rstrip('s'))
            return str(timedelta(seconds=seconds))[:-3]  # Remove microseconds
        parts = timestamp.split(':')
        if len(parts) == 3:
            return timestamp
        elif len(parts) == 2:
            return f"00:{timestamp}"
        else:
            seconds = int(float(timestamp))
            return str(timedelta(seconds=seconds))[:-3]  # Remove microseconds
    except ValueError:
        print(f"Warning: Unable to convert timestamp '{timestamp}'. Keeping original format.")
        return timestamp

def extract_section(transcript, prompt_template):
    prompt = PromptTemplate(template=prompt_template, input_variables=["transcript"])
    query = prompt.format(transcript=transcript)
    response = model.invoke(query)
    return response.content

def post_process_content(content):
    prompt = PromptTemplate(template=post_processing_prompt, input_variables=["content"])
    query = prompt.format(content=content)
    response = model.invoke(query)
    return response.content

def structure_markdown(content, title, base_name, participants):
    structured_content = f"# {base_name}\n\n"
    
    sections = re.split(r'\n(?=(?:Meeting Overview|Main Topics|Decisions and Impact|Action Items))', content)
    for section in sections:
        lines = section.split('\n')
        if lines:
            header = lines[0].strip()
            if header in ['Meeting Overview', 'Main Topics', 'Decisions and Impact', 'Action Items']:
                structured_content += f"## {header}\n"
                for line in lines[1:]:
                    if line.strip():
                        structured_content += f"{line}\n"
            else:
                for line in lines:
                    if line.strip():
                        structured_content += f"{line}\n"
            structured_content += "\n"
    
    structured_content += "## Participants\n"
    for participant in participants:
        structured_content += f"- {participant['name']} (Total Duration: {participant['total_duration']})\n"
    
    return structured_content

def extract_meeting_notes(transcript, participants, title, base_name):
    overview = extract_section(transcript, overview_prompt)
    main_topics = extract_section(transcript, main_topics_prompt)
    decisions_impact = extract_section(transcript, decisions_impact_prompt)
    action_items = extract_section(transcript, action_items_prompt)
    
    content = f"Meeting Overview\n{overview}\n\nMain Topics\n{main_topics}\n\nDecisions and Impact\n{decisions_impact}\n\nAction Items\n{action_items}"
    
    processed_content = post_process_content(content)
    structured_content = structure_markdown(processed_content, title, base_name, participants)
    
    return structured_content

def save_md_to_file(content, file_path):
    directory = os.path.dirname(file_path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    with open(file_path, 'w') as f:
        f.write(content)
    print(f"Markdown file saved to {file_path}")

def process_files(directory, title):
    vtt_files = glob.glob(os.path.join(directory, "*.vtt"))
    for file_path in vtt_files:
        base_name = os.path.basename(file_path).replace(".vtt", "")
        csv_file_path = os.path.join(directory, f"{base_name}.csv")
        participants = read_csv_file(csv_file_path) if os.path.exists(csv_file_path) else []
        transcript = read_vtt_file(file_path)
        result = extract_meeting_notes(transcript, participants, title, base_name)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        md_file_path = os.path.join(directory, "md", f"summary_{base_name}_{timestamp}.md")
        print("\nExtracted Information:")
        if result:
            print(result)
            save_md_to_file(result, md_file_path)
        else:
            print("No valid information extracted.")
        print("Information extraction completed.")

def main():
    parser = argparse.ArgumentParser(description="Extract meeting notes from a transcript file or directory.")
    parser.add_argument("-i", "--input", type=str, help="Path to the input transcript file (VTT format)")
    parser.add_argument("-m", "--output-md", type=str, default=None, help="Path to save the output Markdown file")
    parser.add_argument("-d", "--directory", type=str, help="Path to the directory containing the transcript files (VTT format)")
    parser.add_argument("-t", "--title", type=str, default="Metaverse Standards Forum", help="Title for the meeting summary")
    args = parser.parse_args()

    if args.directory:
        process_files(args.directory, args.title)
    elif args.input:
        file_path = args.input
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return
        base_name = os.path.basename(file_path).replace(".vtt", "")
        csv_file_path = os.path.join(os.path.dirname(file_path), f"{base_name}.csv")
        participants = read_csv_file(csv_file_path) if os.path.exists(csv_file_path) else []
        transcript = read_vtt_file(file_path)
        result = extract_meeting_notes(transcript, participants, args.title, base_name)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        if args.output_md:
            md_file_path = args.output_md
        else:
            md_file_path = os.path.join(os.path.dirname(file_path), "md", f"summary_{base_name}_{timestamp}.md")
        print("\nExtracted Information:")
        if result:
            print(result)
            save_md_to_file(result, md_file_path)
        else:
            print("No valid information extracted.")
        print("Information extraction completed.")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
