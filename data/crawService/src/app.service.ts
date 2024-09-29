import FirecrawlApp from '@mendable/firecrawl-js';
import { Injectable } from '@nestjs/common';
import * as fs from 'fs'; // Import the 'fs' module

const fireCrawl = new FirecrawlApp({
  apiKey: 'fc-460438c8d3b0405887af8fbc0a24d1fc',
});

// https://tuoitre.vn/
@Injectable()
export class AppService {
  public getData = async (urlName: string) => {
    const crawlResult = await fireCrawl.crawlUrl(urlName, {
      limit: 1,
      scrapeOptions: {
        formats: ['markdown'],
      },
    });

    if (!crawlResult.success) {
      if ('error' in crawlResult) {
        throw new Error(`Failed to crawl: ${crawlResult.error}`);
      } else {
        throw new Error('Failed to crawl');
      }
    }
    const jsonData = JSON.stringify(crawlResult, null, 2); // Convert object to JSON string

    // Xuất dữ liệu ra file
    return new Promise<void>((resolve, reject) => {
      fs.writeFile('dataCrawl.json', jsonData, (err) => {
        if (err) {
          reject('Có lỗi khi ghi file: ' + err);
        } else {
          resolve();
          return {
            message: 'Ghi file thành công',
            statusCode: 200,
          };
        }
      });
    });
  };
}
