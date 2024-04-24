import json

def json_to_txt(json_path, txt_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    with open(txt_path, 'w') as f:
        for line in data["original"]:
            f.write(line + '\n')

json_to_txt('./gpt-4/writing_gpt-4.raw_data.json', './gpt-4/writing_gpt-4.txt')
json_to_txt('./gpt-3.5/writing_gpt-3.5-turbo.raw_data.json', './gpt-3.5/writing_gpt-3.5-turbo.txt')
json_to_txt('./davinci/writing_davinci.raw_data.json', './davinci/writing_davinci.txt')