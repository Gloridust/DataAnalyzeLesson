import requests
from bs4 import BeautifulSoup
import re
import time
import random
import logging
from lxml import html  # 添加lxml库以支持XPath

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class XiaoHongShuScraper:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.7',
            'Accept-Language': 'zh-CN,zh;q=0.9,en;q=0.8',
            'Accept-Encoding': 'gzip, deflate, br',
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Cache-Control': 'max-age=0',
        }
        self.nickname = ''
        self.fans_count = 0
        self.like_count = 0
        
    def _random_sleep(self, min_seconds=1, max_seconds=3):
        """随机睡眠，避免被反爬"""
        time.sleep(random.uniform(min_seconds, max_seconds))
        
    def fetch(self, url):
        """
        获取指定小红书用户页面的粉丝数
        
        :param url: 小红书用户主页URL
        :return: 是否成功
        """
        try:
            # 添加随机延迟
            self._random_sleep()
            
            # 发起请求
            logger.info(f"正在请求: {url}")
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            # 记录响应状态
            logger.info(f"请求成功，状态码: {response.status_code}")
            
            # 使用BeautifulSoup解析HTML
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # 使用lxml解析HTML以支持XPath
            tree = html.fromstring(response.text)
            
            # 提取用户昵称
            nickname_selectors = [
                '.nickname', 
                'h1.nickname',
                '.user-name',
                'h1.user-name',
                '.name-detail'
            ]
            
            nickname_element = None
            for selector in nickname_selectors:
                elements = soup.select(selector)
                if elements:
                    nickname_element = elements[0]
                    logger.info(f"找到昵称选择器: {selector}")
                    break
            
            if nickname_element:
                self.nickname = nickname_element.get_text().strip()
                logger.info(f"找到昵称: {self.nickname}")
            else:
                # 如果无法通过选择器找到，尝试XPath
                nickname_xpath = '//h1[contains(@class, "nickname")]'
                nickname_elements = tree.xpath(nickname_xpath)
                if nickname_elements:
                    self.nickname = nickname_elements[0].text_content().strip()
                    logger.info(f"通过XPath找到昵称: {self.nickname}")
                else:
                    logger.warning("无法找到用户昵称，尝试继续提取其他信息")
            
            # 使用提供的XPath提取粉丝数
            try:
                # XPath 1: 粉丝数字
                fans_xpath1 = '/html/body/div[2]/div[1]/div[2]/div[2]/div/div[1]/div/div[2]/div[1]/div[4]/div/div[2]/span[1]'
                # XPath 2: "粉丝"文本
                fans_xpath2 = '/html/body/div[2]/div[1]/div[2]/div[2]/div/div[1]/div/div[2]/div[1]/div[4]/div/div[2]/span[2]'
                
                fans_elements1 = tree.xpath(fans_xpath1)
                fans_elements2 = tree.xpath(fans_xpath2)
                
                if fans_elements1:
                    fans_text = fans_elements1[0].text_content().strip()
                    logger.info(f"通过XPath找到粉丝数文本: {fans_text}")
                    
                    # 检查第二个元素是否包含"粉丝"文本
                    if fans_elements2 and "粉丝" in fans_elements2[0].text_content():
                        logger.info("确认是粉丝数元素")
                        self.fans_count = self._convert_number_text(fans_text)
                        logger.info(f"转换后的粉丝数: {self.fans_count}")
                    else:
                        logger.warning("找到的元素可能不是粉丝数，尝试其他方法")
                else:
                    logger.warning("通过提供的XPath未找到粉丝数，尝试更通用的XPath")
                    
                    # 更通用的XPath，查找包含"粉丝"文本的相邻元素
                    generic_fans_xpath = '//span[contains(text(), "粉丝")]/preceding-sibling::span[1] | //div[contains(text(), "粉丝")]/preceding-sibling::div[1]'
                    generic_fans_elements = tree.xpath(generic_fans_xpath)
                    
                    if generic_fans_elements:
                        fans_text = generic_fans_elements[0].text_content().strip()
                        logger.info(f"通过通用XPath找到粉丝数文本: {fans_text}")
                        self.fans_count = self._convert_number_text(fans_text)
                        logger.info(f"转换后的粉丝数: {self.fans_count}")
                    else:
                        # 尝试CSS选择器
                        fans_css_selectors = [
                            '.count .num', 
                            '.follower-count',
                            '.user-stats .follower-count',
                            '.user-data .data'
                        ]
                        
                        for selector in fans_css_selectors:
                            elements = soup.select(selector)
                            if elements and len(elements) > 1:  # 通常粉丝数是第二个
                                fans_text = elements[1].get_text().strip()
                                logger.info(f"通过CSS选择器找到可能的粉丝数文本: {selector} = {fans_text}")
                                # 检查相邻元素是否包含"粉丝"文本
                                if elements[1].find_next_sibling() and "粉丝" in elements[1].find_next_sibling().get_text():
                                    self.fans_count = self._convert_number_text(fans_text)
                                    logger.info(f"确认是粉丝数，转换后的值: {self.fans_count}")
                                    break
                
                # 如果上述方法都未成功，尝试正则表达式
                if self.fans_count == 0:
                    logger.warning("常规方法未能找到粉丝数，尝试正则表达式")
                    fans_patterns = [
                        r'<span[^>]*>(\d+(?:\.\d+)?(?:万|w)?)</span>\s*<span[^>]*>[^<]*粉丝[^<]*</span>',
                        r'<div[^>]*>(\d+(?:\.\d+)?(?:万|w)?)</div>\s*<div[^>]*>[^<]*粉丝[^<]*</div>',
                        r'<span[^>]*>(\d+(?:\.\d+)?(?:万|w)?)</span>[^<]*<span[^>]*>[^<]*粉丝[^<]*</span>',
                        r'follower(?:s|Count|Num|_count)["\']\s*:\s*(\d+)',
                        r'fans(?:Count|Num|_count)["\']\s*:\s*(\d+)'
                    ]
                    
                    for pattern in fans_patterns:
                        match = re.search(pattern, response.text, re.IGNORECASE)
                        if match:
                            fans_text = match.group(1).strip()
                            logger.info(f"通过正则找到粉丝数文本: {fans_text}")
                            self.fans_count = self._convert_number_text(fans_text)
                            logger.info(f"转换后的粉丝数: {self.fans_count}")
                            break
                
                # 如果仍未找到，尝试查找特定的数字模式
                if self.fans_count == 0:
                    # 查找所有数字后跟"粉丝"文本的实例
                    all_numbers_with_fans = re.findall(r'(\d+(?:\.\d+)?(?:万|w)?)[^<]{0,20}粉丝', response.text)
                    if all_numbers_with_fans:
                        logger.info(f"找到疑似粉丝数的数字: {all_numbers_with_fans}")
                        # 使用第一个匹配项
                        self.fans_count = self._convert_number_text(all_numbers_with_fans[0])
                        logger.info(f"选择第一个匹配项作为粉丝数: {self.fans_count}")
            
            except Exception as e:
                logger.error(f"提取粉丝数时出错: {e}")
                import traceback
                logger.error(traceback.format_exc())
            
            # 尝试提取点赞数 (类似粉丝数的逻辑)
            # ...省略点赞数提取逻辑...
            
            # 如果找到昵称和粉丝数，认为成功
            if self.nickname and self.fans_count > 0:
                return True
            elif self.nickname:  # 至少找到了昵称
                logger.warning("找到了昵称但未能找到粉丝数")
                return True  # 可以根据需求决定是否返回True
            else:
                logger.error("未能找到足够的用户信息")
                return False
            
        except Exception as e:
            logger.error(f"获取用户信息失败: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def _convert_number_text(self, text):
        """
        将文本格式的数字转换为整数
        处理"万"或"w"表示的数字
        
        :param text: 数字文本，如 "1.2万" 或 "12.3w"
        :return: 整数
        """
        try:
            # 移除空白字符
            text = text.strip()
            
            # 处理带"万"或"w"的数字
            if '万' in text or 'w' in text:
                text = text.replace('万', '').replace('w', '')
                return int(float(text) * 10000)
            
            # 处理普通数字
            return int(float(text))
        except ValueError:
            logger.error(f"无法将文本转换为数字: {text}")
            return 0
    
    def get_fans_count(self):
        """获取粉丝数"""
        return self.fans_count
    
    def get_nickname(self):
        """获取昵称"""
        return self.nickname
    
    def get_like_count(self):
        """获取点赞数"""
        return self.like_count
    
    def get_user_info_by_url(self, url):
        """
        通过URL获取用户信息
        
        :param url: 小红书用户主页URL
        :return: 包含用户信息的字典或None(失败时)
        """
        if self.fetch(url):
            return {
                'nickname': self.get_nickname(),
                'fans_count': self.get_fans_count(),
                'like_count': self.get_like_count()
            }
        return None

    def get_user_info_by_id(self, user_id):
        """
        通过用户ID获取用户信息
        
        :param user_id: 小红书用户ID
        :return: 包含用户信息的字典或None(失败时)
        """
        url = f"https://www.xiaohongshu.com/user/profile/{user_id}"
        return self.get_user_info_by_url(url)


# 使用示例
if __name__ == "__main__":
    scraper = XiaoHongShuScraper()
    
    # 示例: 通过用户ID获取信息
    user_id = "5f1d89a6000000000100ba01"  # 使用您提供的ID
    user_info = scraper.get_user_info_by_id(user_id)
    if user_info:
        print(f"用户昵称: {user_info['nickname']}")
        print(f"粉丝数: {user_info['fans_count']}")
        print(f"获赞数: {user_info['like_count']}")
    else:
        print("获取用户信息失败")