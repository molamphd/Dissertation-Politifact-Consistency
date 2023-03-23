import requests
import pandas as pd
import csv
from bs4 import BeautifulSoup
from random import randint
from time import sleep
import re

# Func
def get_topic_urls():
    headers = {
        'Access-Control-Allow-Origin': '*',
        'Access-Control-Allow-Methods': 'GET',
        'Access-Control-Allow-Headers': 'Content-Type',
        'accept': '*/*',
        'accept-encoding': 'gzip, deflate',
        'accept-language': 'en,mr;q=0.9',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.75 Safari/537.36'
    }
    url = 'https://www.politifact.com/issues/'
    req = requests.get(url, headers=headers)
    soup = BeautifulSoup(req.content, 'html.parser')
    html = soup.find_all('div', class_="c-chyron__value")
    topics = []
    count = 0
    with open('Politifact-Links.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        for result in html:
            print(f'Result Index = {count}')
            a = result.find('a', href=True)
            topic = a['href'].replace('/', '')
            topic_url = f'https://www.politifact.com/factchecks/list/?category={topic}'
            writer.writerow([topic_url])
            multiple_pages = get_multiple_pages(topic_url)
            writer.writerows(multiple_pages)
            count = count + 1
        file.close()
    return topics


def get_header_and_ratings():
    headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.1.2222.33 Safari/537.36",
                "Accept-Encoding": "*",
                "Connection": "keep-alive"
    }

    count = 0

    df = pd.read_csv("Politifact-Links.csv", header=None)
    urls = df.iloc[:, 0].tolist()

    with open('Updated_Politifacts.csv', 'w', newline='', encoding="utf-8") as file:
        writer = csv.writer(file)
        for url in urls:
            if count > 625:
                print(f'URL {count} processing now')
                html2 = requests.get(url, headers=headers)
                sleep(randint(2, 7))
                soup = BeautifulSoup(html2.content, 'html.parser')

                sourcesHtml = soup.find_all('a', class_="m-statement__name")
                headersHtml = soup.find_all('div', class_="m-statement__quote")
                ratingsHtml = soup.find_all('img', class_="c-image__original", alt=True)

                for sourceResult, headersResult, ratingsResult in zip(sourcesHtml, headersHtml, ratingsHtml):
                    factCheckUrl = f'https://www.politifact.com{headersResult.find("a")["href"]}'
                    judgement_text, judgement_author, text_found = get_judgement_data(factCheckUrl)
                    if text_found == True:
                        header = headersResult.text.replace('\n\n', '')
                        source = sourceResult['title']
                        author_url = f'https://www.politifact.com{sourceResult["href"]}'
                        author_bio = get_author_bio(author_url)
                        rating = ratingsResult['alt']
                        row = [source, header, author_bio, judgement_text, judgement_author, factCheckUrl, rating]
                        writer.writerow(row)
            count = count + 1
        file.close()


def get_author_bio(url):
    headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.1.2222.33 Safari/537.36",
                "Accept-Encoding": "*",
                "Connection": "keep-alive"
    }
    req = requests.get(url, headers=headers)
    sleep(randint(2, 7))
    soup = BeautifulSoup(req.content, 'html.parser')
    for author_bio in soup.find('div', attrs={'class': 'm-pageheader__body'}).find_all('p'):
        result = author_bio.text
    return result


def get_multiple_pages(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.1.2222.33 Safari/537.36",
        "Accept-Encoding": "*",
        "Connection": "keep-alive"
    }

    multiple_page_urls = []
    while url:
        req = requests.get(url, headers=headers)
        sleep(randint(2, 7))
        soup = BeautifulSoup(req.content, 'html.parser')
        buttons = soup.findAll('a', class_="c-button c-button--hollow")
        prev_url = url
        for button in buttons:
            if button.text == 'Next':
                url = f'https://www.politifact.com/factchecks/list/{button["href"]}'
                multiple_page_urls.append([url])
        if prev_url == url:
            break
    return multiple_page_urls


def get_judgement_data(url):
    headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/71.1.2222.33 Safari/537.36",
                "Accept-Encoding": "*",
                "Connection": "keep-alive"
    }
    req = requests.get(url, headers=headers)
    sleep(randint(2, 7))
    soup = BeautifulSoup(req.content, 'html.parser')
    judgement_text = ''
    truth_rating_array = ['true', 'mostly true', 'half true', 'mostly false', 'false', 'pants on fire']
    div_tags = soup.find_all('div', attrs={'class': 'pf_subheadline'})
    text_found = False
    for div_tag in div_tags:
        if 'our ruling' in div_tag.text.lower() or 'our rating' in div_tag.text.lower() or 'our conclusion' in div_tag.text.lower():
            p_tags = div_tag.find_next_siblings("p")
            for p in p_tags:
                stripped_text = p.text.replace(".", "")
                stripped_text = stripped_text.replace(",", "")
                stripped_text = stripped_text.replace("!", "")
                stripped_text - stripped_text.lower()
                if any(truth_rating in stripped_text for truth_rating in truth_rating_array):
                    break
                else:
                    judgement_text = judgement_text + f'{p.text} '
                    text_found = True
    else:
        article_tag = soup.find('article', attrs={'class': 'm-textblock'})
        if article_tag:
            p_tags = article_tag.find_all('p')
            if len(p_tags) > 5:
                for p in p_tags:
                    if 'our ruling' in p.text.lower() or 'our rating' in p.text.lower() or 'our conclusion' in p.text.lower():
                        p2_tags = p.find_next_siblings("p")
                        for p2 in p2_tags:
                            stripped_text = p2.text.replace(".", "")
                            stripped_text = stripped_text.replace(",", "")
                            stripped_text = stripped_text.replace("!", "")
                            if any(truth_rating in stripped_text for truth_rating in truth_rating_array):
                                break
                            else:
                                judgement_text = judgement_text + f'{p2.text} '
                                text_found = True
    try:
        author_link = soup.find('div', attrs={'class': 'm-author__content copy-xs u-color--chateau'}).find('a')
        judgement_author = author_link.text if author_link else "N/A"
    except AttributeError:
        judgement_author = "N/A"
    return judgement_text, judgement_author, text_found


if __name__ == '__main__':
    # topicUrls = get_topic_urls()
    get_header_and_ratings()
