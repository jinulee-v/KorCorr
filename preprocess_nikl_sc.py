import json

with open("data/EXSC2102112091.json", "r", encoding="UTF-8") as file:
    edata = json.load(file)
with open("data/MXSC2102112091.json", "r", encoding="UTF-8") as file:
    mdata = json.load(file)

data = []
for dataset in [edata, mdata]:
    for document in dataset["document"]:
        # edata and mdata have different scheme
        if "paragraph" in document:
            key = "paragraph"
        else:
            key = "utterance"

        for para in document[key]:
            form = para["form"]
            corr = para["corrected_form"]
            if len(form) < 5 or len(corr) < 5:
                # Skip too short sentences
                continue
            data.append({
                "form": form,
                "corrected_form": corr
            })
        
with open("data/nikl_sc.json", "w", encoding="UTF-8") as file:
    json.dump(data, file, indent=4, ensure_ascii=False)