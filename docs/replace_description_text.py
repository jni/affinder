"""
for ../.napari/DESCRIPTION.md
find URLs ending in .mp4 and replace URL wrapped in HTML video tag 
"""

replacement_url = '<video width="640" height="480" controls>\n\
    <source src="{}" type="video/mp4">\n\
Your browser does not support the video tag.\n\
</video>\n'
new_text = ""

with open(".napari/DESCRIPTION.md", 'r+') as desc_file:
    for line in desc_file:
        if line.strip().startswith("https://user-images.githubusercontent") and line.strip().endswith(".mp4"):
            line = replacement_url.format(line.strip())
        new_text += line
    desc_file.seek(0)
    desc_file.write(new_text)
