"""
for ../.napari/DESCRIPTION.md
find URLs ending in .mp4 and replace URL wraped in HTML video tag 
"""

desc_source = ".napari/DESCRIPTION.md"
desc_file = open(desc_source, 'r')
replacement_url = '<video width="640" height="480" controls>\n\
    <source src="{}" type="video/mp4">\n\
Your browser does not support the video tag.\n\
</video>\n'
new_text = ""

for line in desc_file:
    if line.strip().startswith("https://user-images.githubusercontent") and line.strip().endswith(".mp4"):
        line = replacement_url.format(line.strip())
    new_text += line
desc_file.close()

desc_file = open(desc_source, 'w')
desc_file.write(new_text)
desc_file.close()
