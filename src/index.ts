import { createOpenAICompatible } from '@ai-sdk/openai-compatible';
import { generateObject, generateText, stepCountIs, streamText, tool } from 'ai';
import { z } from 'zod';

const productSchema = z.object({
	name: z.string(),
	price: z.number(),
	currency: z.string(),
	inStock: z.boolean(),
	dimensions: z.object({
		length: z.number(),
		width: z.number(),
		height: z.number(),
		unit: z.string(),
	}),
	manufacturer: z.object({
		name: z.string(),
		country: z.string(),
		website: z.string(),
	}),
	specifications: z.object({
		weight: z.number(),
		weightUnit: z.string(),
		warrantyMonths: z.number(),
	}),
});

const INTERACTION_ID_HEADER = 'X-Interaction-Id';

type ChatMessage = { role: 'user' | 'assistant'; content: string };
const conversationHistory = new Map<string, ChatMessage[]>();

export default {
	async fetch(request: Request, env: Env): Promise<Response> {
		const url = new URL(request.url);

		if (request.method !== 'POST' || url.pathname !== '/api') {
			return new Response('Not Found', { status: 404 });
		}

		const challengeType = url.searchParams.get('challengeType');
		if (!challengeType) {
			return new Response('Missing challengeType query parameter', {
				status: 400,
			});
		}

		const interactionId = request.headers.get(INTERACTION_ID_HEADER);
		if (!interactionId) {
			return new Response(`Missing ${INTERACTION_ID_HEADER} header`, {
				status: 400,
			});
		}

		const payload = await request.json<any>();

		switch (challengeType) {
			case 'HELLO_WORLD':
				return Response.json({
					greeting: `Hello ${payload.name}`,
				});
			case 'BASIC_CONVERSATION_AND_MEMORY': {
				if (!env.DEV_SHOWDOWN_API_KEY) {
					throw new Error('DEV_SHOWDOWN_API_KEY is required');
				}

				const history = conversationHistory.get(interactionId) ?? [];
				history.push({ role: 'user', content: payload.message });

				const workshopLlm = createWorkshopLlm(env.DEV_SHOWDOWN_API_KEY, interactionId);
				const result = await generateText({
					model: workshopLlm.chatModel('passo-2.5'),
					system:
						'You are a helpful assistant engaged in a multi-turn conversation about product details. ' +
						'Users will share facts about products (prices, warranties, dimensions, etc.) over several turns, then ask you to compute totals or derived values based on everything shared so far. ' +
						'Remember all details from previous messages in this conversation. ' +
						'When asked for totals, show your arithmetic reasoning briefly and then give a concise natural-language answer that states the requested values (with units).',
					messages: history,
				});

				const answer = result.text || 'N/A';
				history.push({ role: 'assistant', content: answer });
				conversationHistory.set(interactionId, history);

				return Response.json({ answer });
			}
			case 'BASIC_LLM': {
				if (!env.DEV_SHOWDOWN_API_KEY) {
					throw new Error('DEV_SHOWDOWN_API_KEY is required');
				}

				const workshopLlm = createWorkshopLlm(env.DEV_SHOWDOWN_API_KEY, interactionId);
				const result = await generateText({
					model: workshopLlm.chatModel('deli-4'),
					system: 'You are a trivia question player. Answer the question correctly and concisely.',
					prompt: payload.question,
				});

				return Response.json({
					answer: result.text || 'N/A',
				});
			}
			case 'BASIC_TOOL_CALL': {
				if (!env.DEV_SHOWDOWN_API_KEY) {
					throw new Error('DEV_SHOWDOWN_API_KEY is required');
				}

				const workshopLlm = createWorkshopLlm(env.DEV_SHOWDOWN_API_KEY, interactionId);
				const result = await generateText({
					model: workshopLlm.chatModel('deli-4'),
					system:
						'You are a weather assistant. Use the getWeather tool to look up the current weather for the city mentioned in the question, then answer in one short natural-language sentence that includes the temperature.',
					prompt: payload.question,
					stopWhen: stepCountIs(5),
					tools: {
						getWeather: tool({
							description: 'Get the current weather for a given city.',
							inputSchema: z.object({
								city: z.string().describe('The city to get the weather for'),
							}),
							execute: async ({ city }) => {
								const res = await fetch('https://devshowdown.com/api/weather', {
									method: 'POST',
									headers: {
										'Content-Type': 'application/json',
										[INTERACTION_ID_HEADER]: interactionId,
									},
									body: JSON.stringify({ city }),
								});
								if (!res.ok) {
									return { error: `Weather API returned ${res.status}` };
								}
								return await res.json();
							},
						}),
					},
				});

				return Response.json({
					answer: result.text || 'N/A',
				});
			}
			case 'MULTI_DOCUMENT_KB_RETRIEVAL': {
				if (!env.DEV_SHOWDOWN_API_KEY) {
					throw new Error('DEV_SHOWDOWN_API_KEY is required');
				}

				const workshopLlm = createWorkshopLlm(env.DEV_SHOWDOWN_API_KEY, interactionId);
				const result = await generateText({
					model: workshopLlm.chatModel('passo-2.5'),
					system:
						'You answer questions about tech conferences, speakers, and events by calling the searchKnowledgeBase tool. ' +
						'IMPORTANT — the tool is LITERAL case-insensitive SUBSTRING grep (like ripgrep), NOT semantic search. ' +
						'A query only matches documents that contain that exact substring verbatim. ' +
						'Rules for queries:\n' +
						'- Prefer ONE short, distinctive token per search: a proper noun, a handle (e.g. @dachfest), a product name, or a conference name.\n' +
						'- NEVER use multi-word descriptive phrases like "closing keynote about technology" — those will almost never match.\n' +
						'- If a query returns 0 results, try a SHORTER or DIFFERENT single token (e.g. drop adjectives, try the conference name alone, try a plausible speaker first name).\n' +
						'- If a query returns too many results, add ONE more distinctive token only if you know it appears verbatim.\n' +
						'- Up to 10 results are returned per search. Read them all before deciding the next step.\n' +
						'Issue as many searches as needed. Once the relevant document(s) are identified, answer in one concise factual sentence grounded in the retrieved text.',
					prompt: payload.question,
					stopWhen: stepCountIs(10),
					tools: {
						searchKnowledgeBase: tool({
							description:
								'Grep-like search over the knowledge base. Case-insensitive substring match. Returns up to 10 documents.',
							inputSchema: z.object({
								query: z.string().describe('The substring to search for in the knowledge base.'),
							}),
							execute: async ({ query }) => {
								const res = await fetch('https://devshowdown.com/api/kb/search', {
									method: 'POST',
									headers: {
										'Content-Type': 'application/json',
										[INTERACTION_ID_HEADER]: interactionId,
									},
									body: JSON.stringify({ query }),
								});
								if (!res.ok) {
									return { error: `Search API returned ${res.status}` };
								}
								return await res.json();
							},
						}),
					},
				});

				return Response.json({
					answer: result.text || 'N/A',
				});
			}
			case 'SINGLE_DOCUMENT_KB_RETRIEVAL': {
				if (!env.DEV_SHOWDOWN_API_KEY) {
					throw new Error('DEV_SHOWDOWN_API_KEY is required');
				}

				const docRes = await fetch('https://devshowdown.com/api/kb/document', {
					headers: {
						[INTERACTION_ID_HEADER]: interactionId,
					},
				});
				if (!docRes.ok) {
					return new Response(`Failed to fetch knowledge base document: ${docRes.status}`, {
						status: 502,
					});
				}
				const document = await docRes.text();

				const workshopLlm = createWorkshopLlm(env.DEV_SHOWDOWN_API_KEY, interactionId);
				const result = await generateText({
					model: workshopLlm.chatModel('deli-4'),
					system:
						'You answer questions strictly based on the provided knowledge base document. Be concise and factual. If the answer is not in the document, say so.',
					prompt: `Knowledge base document:\n\n${document}\n\nQuestion: ${payload.question}`,
				});

				return Response.json({
					answer: result.text || 'N/A',
				});
			}
			case 'RESPONSE_STREAMING': {
				if (!env.DEV_SHOWDOWN_API_KEY) {
					throw new Error('DEV_SHOWDOWN_API_KEY is required');
				}

				const workshopLlm = createWorkshopLlm(env.DEV_SHOWDOWN_API_KEY, interactionId);
				const result = streamText({
					model: workshopLlm.chatModel('deli-4'),
					prompt: payload.prompt,
				});

				return result.toTextStreamResponse();
			}
			case 'JSON_MODE': {
				if (!env.DEV_SHOWDOWN_API_KEY) {
					throw new Error('DEV_SHOWDOWN_API_KEY is required');
				}

				const workshopLlm = createWorkshopLlm(env.DEV_SHOWDOWN_API_KEY, interactionId);
				const result = await generateObject({
					model: workshopLlm.chatModel('deli-4'),
					schema: productSchema,
					system:
						'Extract product information from the given description and return it as a JSON object matching the provided schema. Preserve numeric values exactly as stated in the text.',
					prompt: payload.description,
				});

				return Response.json(result.object);
			}
				default:
					return new Response('Solver not found', { status: 404 });
			}
		},
	} satisfies ExportedHandler<Env>;

function createWorkshopLlm(apiKey: string, interactionId: string) {
	return createOpenAICompatible({
		name: 'dev-showdown',
		baseURL: 'https://devshowdown.com/v1',
		supportsStructuredOutputs: true,
		headers: {
			Authorization: `Bearer ${apiKey}`,
			[INTERACTION_ID_HEADER]: interactionId,
		},
	});
}
