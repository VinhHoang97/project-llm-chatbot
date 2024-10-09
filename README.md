### How to run it on your own machine

1. Install the requirements

   ```
   $ docker-compose build
   ```

2. Run the app

   ```
   $ docker-compose up
   $ docker-compose exec ollama ollama pull llama3.2:1b
   ```

3. Cách chạy crawService:

- cd vào folder crawService
- pnpm i (nếu chạy lần đầu)
- pnpm run start:dev
- run dataCrawl.py to format data
