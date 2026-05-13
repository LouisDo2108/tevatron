from wikimapper import WikiMapper

"""
# Download the database
wikimapper download enwiki-20220820 --mirror https://dumps.wikimedia.your.org --dir /home/thuy0050/mg61_scratch2/thuy0050/data/third_work/flashrag

# Create the mapping index
wikimapper create enwiki-20220820 --dumpdir /home/thuy0050/mg61_scratch2/thuy0050/data/third_work/flashrag --target /home/thuy0050/mg61_scratch2/thuy0050/data/third_work/flashrag/index_enwiki-20220820.db
"""

mapper = WikiMapper("/home/thuy0050/mg61_scratch2/thuy0050/data/third_work/flashrag/index_enwiki-20220820.db")

# Map Wikidata id to Wikipedia id
page_title = mapper.wikipedia_id_to_title(12)
print(page_title) 

# Map Wikidata id to Wikipedia id
wikipedia_ids = mapper.id_to_wikipedia_ids("Q134430")
page_title = mapper.wikipedia_id_to_title(wikipedia_ids[0])


print(mapper.title_to_id("Carl_Eric_Almgren")) # Q5040099
print(mapper.url_to_id("/wiki/Carl_Eric_Almgren#P39#1")) # None