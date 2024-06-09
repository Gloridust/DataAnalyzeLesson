import requests
from bs4 import BeautifulSoup
import pandas as pd
import time

def get_amazon_reviews(url, num_pages):
    reviews = []
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"}

    for page in range(1, num_pages+1):
        page_url = f"{url}&pageNumber={page}"
        response = requests.get(page_url, headers=headers)
        soup = BeautifulSoup(response.content, 'html.parser')

        review_divs = soup.find_all('div', {'data-hook': 'review'})
        for review in review_divs:
            review_text = review.find('span', {'data-hook': 'review-body'}).text.strip()
            reviews.append(review_text)
        
        time.sleep(1)  # 防止过快请求被封禁

    return reviews

# 替换为你想要爬取的亚马逊产品评论页面URL
product_url = "https://amazon.sg/Hashun-Insulated-Leak-Proof-Stainless-Reusable/dp/B0BJ5YKL5X/ref=srd_d_ssims_T1_d_sccl_1_1/356-1914499-4201127?pd_rd_w=txFHX&content-id=amzn1.sym.11fe6edc-934b-4ea1-a57c-6d93e120f36b&pf_rd_p=11fe6edc-934b-4ea1-a57c-6d93e120f36b&pf_rd_r=A0JGED73TEWFSKPDMPA0&pd_rd_wg=nzukg&pd_rd_r=4ab44cb2-997d-4d85-8c72-e3b03705709e&pd_rd_i=B0BJ5YKL5X&psc=1"
reviews = get_amazon_reviews(product_url, 10)  # 爬取10页评论

# 保存到CSV文件
df = pd.DataFrame(reviews, columns=['Review'])
df.to_csv('amazon_reviews.csv', index=False)
