# Dev Showdown Cloudflare Starter

This starter is for [Dev Showdown](https://devshowdown.com) workshop participants using Cloudflare Workers.

[![Deploy to Cloudflare](https://deploy.workers.cloudflare.com/button)](https://deploy.workers.cloudflare.com/?url=https://github.com/cowchimp/dev-showdown-cloudflare-starter)

## Fastest Start

Click the Deploy to Cloudflare button above.

Cloudflare will:

- clone the repo into your GitHub account
- create and deploy your Worker
- prompt you for the required secret during setup

The required secret is:

- `DEV_SHOWDOWN_API_KEY`

Get that from the Dev Showdown workshop Settings page.

## What This Worker Exposes

- `POST /api?challengeType=HELLO_WORLD`
- `POST /api?challengeType=BASIC_LLM`

That is enough to start earning points quickly and gives you a clean base to extend for later challenges.

## Local Development

If you want to work locally instead of deploying first:

1. Create your local env file:

```bash
cp .env.example .env
```

2. Put your `DEV_SHOWDOWN_API_KEY` into `.env`.

3. Install dependencies and start the worker:

```bash
npm install
npm run dev
```
