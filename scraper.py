import wikipediaapi

def scrape_wiki_page(page_title):
    user_agent = "WikiRAGApp/1.0 (prakharag99@gmail.com)"  # Replace with your contact email
    wiki_wiki = wikipediaapi.Wikipedia(
        language='en',
        extract_format=wikipediaapi.ExtractFormat.WIKI,
        user_agent=user_agent
    )
    page = wiki_wiki.page(page_title)
    return page.text

if __name__ == "__main__":
    text = scrape_wiki_page('Luke Skywalker')
    with open('luke_skywalker.txt', 'w') as f:
        f.write(text)
