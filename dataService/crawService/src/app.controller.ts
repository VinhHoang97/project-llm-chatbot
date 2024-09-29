import { Controller, Get, Query } from '@nestjs/common';
import { AppService } from './app.service';

@Controller()
export class AppController {
  constructor(private readonly appService: AppService) {}

  @Get('data')
  getData(@Query('urlName') urlName: string) {
    return this.appService.getData(urlName);
  }
}
