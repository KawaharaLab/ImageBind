import json

import pandas as pd

def sanitize(text: str) -> str:
    """改行をスペースに置き換え、前後の空白を除去"""
    return text.replace('\r', ' ').replace('\n', ' ').strip()

with open("data/filtered_uids.txt", "r") as f:
    content = f.read()
    imu_video_uids = content.split()

with open("data/ego4d.json", "r") as f:
    data = json.load(f)
    uid_scenario_dict = {}
    scenarios_set = set()
    for video in data['videos']:
        video_uid = video['video_uid']
        scenarios = video['scenarios']
        if video_uid not in imu_video_uids:
            continue
        if len(scenarios) != 1:
            continue
        print(f"Video UID {video_uid} has a single scenario: {len(scenarios)} scenario")
        uid_scenario_dict[video_uid] = sanitize(scenarios[0])
        scenarios_set.add(sanitize(scenarios[0]))


uid_scenario_df = pd.DataFrame.from_dict(uid_scenario_dict, orient='index', columns=['scenario'])
uid_scenario_df.index.name = 'video_uid'
uid_scenario_df.to_csv("data/uid_scenario_imu.csv", index=True)
with open("data/uid_scenario_imu.txt", "w") as f:
    for scenario in scenarios_set:
        f.write(f"{sanitize(scenario)}\n")
