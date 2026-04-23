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
				const model = workshopLlm.chatModel('deli-4');

				const actionSchema = z.discriminatedUnion('action', [
					z.object({
						action: z.literal('search'),
						query: z.string().describe('Distinctive substring to grep for in the knowledge base.'),
					}),
					z.object({
						action: z.literal('answer'),
						answer: z.string().describe('The final answer to the user question.'),
					}),
				]);

				const system =
					'You answer questions about tech conferences, speakers, and events by searching a knowledge base of 1667 short documents. ' +
					'You must respond with a JSON object choosing one of two actions: "search" (to run a case-insensitive substring grep, returns up to 10 docs) or "answer" (only when you have enough evidence). ' +
					'Strategy: pick distinctive keywords from the question (proper nouns, uncommon phrases, conference names). ' +
					'If a search returns 0 results, broaden or rephrase (try a single keyword, or a different noun). ' +
					'If a search returns many vague results, narrow with a more specific phrase. ' +
					'Run multiple searches as needed. Once the relevant document(s) are clearly identified, emit action "answer" with a concise factual reply grounded in the documents.';

				const messages: { role: 'user' | 'assistant'; content: string }[] = [
					{ role: 'user', content: payload.question },
				];

				let finalAnswer = 'N/A';
				for (let step = 0; step < 8; step++) {
					const { object } = await generateObject({
						model,
						schema: actionSchema,
						system,
						messages,
					});

					if (object.action === 'answer') {
						finalAnswer = object.answer;
						break;
					}

					const searchRes = await fetch('https://devshowdown.com/api/kb/search', {
						method: 'POST',
						headers: {
							'Content-Type': 'application/json',
							[INTERACTION_ID_HEADER]: interactionId,
						},
						body: JSON.stringify({ query: object.query }),
					});
					const searchJson = searchRes.ok
						? await searchRes.json()
						: { results: [], error: `Search API returned ${searchRes.status}` };

					messages.push({ role: 'assistant', content: JSON.stringify(object) });
					messages.push({
						role: 'user',
						content: `Search results for "${object.query}":\n${JSON.stringify(searchJson)}\n\nEither search again with a refined query or emit your final answer.`,
					});
				}

				return Response.json({ answer: finalAnswer });
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
