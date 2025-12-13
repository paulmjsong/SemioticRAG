import json

def main():
    src_path = "../example/dataset/dataset_emuseum_0F_664.json"
    with open(src_path, "r", encoding="utf-8") as src_file:
        data = json.load(src_file)
    
    new_data = {}
    idx = 1
    
    for key, item in data.items():
        flags = [False, False, False]
        for sentence in item["sentences"]:
            if sentence["labels"][0] == "meaning":
                flags[0] = True
            elif sentence["labels"][0] == "composition":
                flags[1] = True
            # elif sentence["labels"][0] == "technique":
            #     flags[1] = True
            elif sentence["labels"][0] == "context":
                flags[2] = True
        if not flags[0]:
            continue
        if not flags[1]:
            continue
        # if not flags[2]:
        #     continue
        new_data[idx] = item
        idx += 1
    
    dst_path = "../example/dataset/dataset_emuseum.json"
    with open(dst_path, "w", encoding="utf-8") as dst_file:
        json.dump(new_data, dst_file, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    main()