import asyncio
import aiohttp

# 1. Basic Asynchronous Function:
'''
In this example, say_hello is an asynchronous function  
that uses await to simulate a non-blocking operation (sleeping for 1 second). 
The main function calls say_hello, and the program prints "Hello" and "World" with a delay.
'''
async def say_hello():
    await asyncio.sleep(1)  # Simulate a non-blocking operation  (sleeping for 1 second).
    print("Hello")

async def main():
    await say_hello()
    print("World")

asyncio.run(main())

# 2. Concurrent Execution:
'''
In this example, count_up and count_down are two asynchronous functions 
that count up and down with a 1-second delay between each count. 
The main function uses asyncio.gather to execute both functions concurrently, 
resulting in interleaved output.
'''
async def count_up():
    for i in range(1,6):
        await asyncio.sleep(1)
        print(f'Counting up: {i}')

async def cound_down():
    for i in range(5,0,-1):
        await asyncio.sleep(1)
        print(f'Counting down: {i}')

async def main():
    await asyncio.gather(count_up(), cound_down())

asyncio.run(main())

# 3. Fetching URLs Concurrently:
'''
In this example, fetch_url is an asynchronous function that uses the aiohttp library 
to fetch the content of URLs concurrently. 
The main function creates a list of tasks to fetch multiple URLs in parallel using asyncio.gather.
'''
async def fetch_url(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

async def main():
    urls = ["https://openai.com", "https://fb.com"]
    tasks = [fetch_url(url) for url in urls]
    results = await asyncio.gather(*tasks)
    for url, result in zip(urls, results):
        print(f"URL: {url}, Content Length: {len(result)}")

asyncio.run(main())
