exclude_uids = {
        "ea27cc27-037f-4c63-b418-faea630faf8e",
        "2e96ebef-240a-4c8f-8d75-405d2d671021",
        "4a08e95b-cb52-4621-b679-87f617893e19",
        "3d67bdde-232a-442e-a29d-56f8d1a323bf",
        "a3beb693-7e2a-4d6f-8658-88d92b453d57"
        }

input_file = "data/uids.txt"
output_file = "data/uids.txt"

with open(input_file, "r") as f:
    content = f.read()

    video_uids = content.split()

    filtered_uids = [uid for uid in video_uids if uid not in exclude_uids]

    filtered_content = " ".join(filtered_uids)

    with open(output_file, "w") as f:
        f.write(filtered_content)
