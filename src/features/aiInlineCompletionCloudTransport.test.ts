import assert from 'node:assert/strict'
import test from 'node:test'
import {
  CloudCompletionError,
  CloudInlineCompletionClient,
  HOST_CODE_COMPLETION_EVENT_CHANNEL,
  HOST_CODE_COMPLETION_REQUEST_CHANNEL,
  ParentCodeCompletionTransport,
  TextCompletionSseDecoder,
  type CloudCompletionFeedback,
  type CloudCompletionInput,
  type CloudCompletionRequest,
  type CloudCompletionResult,
  type CloudCompletionTransport,
  type HostCodeCompletionEventMessage
} from './aiInlineCompletionCloudTransport'

test('SSE decoder combines arbitrary chunks and preserves code whitespace', () => {
  const decoder = new TextCompletionSseDecoder()

  decoder.push('data: {"id":"cmp_1","choices":[{"text":"  Serial","finish_reason":null}]}\n')
  decoder.push('\ndata: {"id":"cmp_1","choices":[{"text":".begin();\\n","finish_reason":"stop"}]}\n\n')
  decoder.push('data: [DONE]\n\n')

  assert.deepEqual(decoder.finish(), {
    text: '  Serial.begin();\n',
    completionId: 'cmp_1',
    done: true
  })
})

class FakeMessageHost {
  readonly sent: unknown[] = []
  readonly parent = {
    postMessage: (message: unknown) => {
      this.sent.push(message)
    }
  }
  private listener?: (event: MessageEvent) => void

  addEventListener(_type: 'message', listener: (event: MessageEvent) => void): void {
    this.listener = listener
  }

  removeEventListener(_type: 'message', listener: (event: MessageEvent) => void): void {
    if (this.listener === listener) {
      this.listener = undefined
    }
  }

  emit(message: HostCodeCompletionEventMessage): void {
    this.listener?.({ source: this.parent, data: message } as unknown as MessageEvent)
  }
}

function cloudRequest(opportunityId = '11111111-1111-4111-8111-111111111111'): CloudCompletionRequest {
  return {
    opportunityId,
    triggerKind: 'automatic',
    document: { languageId: 'cpp', relativePath: 'src/main.cpp', version: 1 },
    position: { line: 0, character: 7 },
    prefix: 'Serial.',
    suffix: '\n}',
    context: [],
    capabilities: { stream: true, partialAccept: true },
    client: { name: 'aily-coder', version: '0.1.2', sessionId: 'device-session' }
  }
}

test('parent transport sends structured prefix/suffix and parses streamed response', async () => {
  const host = new FakeMessageHost()
  const transport = new ParentCodeCompletionTransport(host, 1_000)
  const request = cloudRequest()
  const completion = transport.complete(request, {})

  assert.deepEqual(host.sent[0], {
    channel: HOST_CODE_COMPLETION_REQUEST_CHANNEL,
    operation: 'complete',
    requestId: request.opportunityId,
    payload: request
  })
  host.emit({
    channel: HOST_CODE_COMPLETION_EVENT_CHANNEL,
    requestId: request.opportunityId,
    type: 'response',
    status: 200,
    headers: { 'X-Aily-Completion-ID': 'cmp_1' }
  })
  host.emit({
    channel: HOST_CODE_COMPLETION_EVENT_CHANNEL,
    requestId: request.opportunityId,
    type: 'chunk',
    chunk: 'data: {"id":"cmp_1","choices":[{"text":"begin();","finish_reason":"stop"}]}\n\n'
  })
  host.emit({
    channel: HOST_CODE_COMPLETION_EVENT_CHANNEL,
    requestId: request.opportunityId,
    type: 'chunk',
    chunk: 'data: [DONE]\n\n'
  })
  host.emit({
    channel: HOST_CODE_COMPLETION_EVENT_CHANNEL,
    requestId: request.opportunityId,
    type: 'end'
  })

  assert.deepEqual(await completion, {
    text: 'begin();',
    completionId: 'cmp_1',
    opportunityId: request.opportunityId
  })
  transport.dispose()
})

test('parent transport propagates cancellation to the authenticated host', async () => {
  const host = new FakeMessageHost()
  const transport = new ParentCodeCompletionTransport(host, 1_000)
  const controller = new AbortController()
  const request = cloudRequest()
  const completion = transport.complete(request, { signal: controller.signal })

  controller.abort()

  await assert.rejects(completion, { name: 'AbortError' })
  assert.deepEqual(host.sent.at(-1), {
    channel: HOST_CODE_COMPLETION_REQUEST_CHANNEL,
    operation: 'cancel',
    requestId: request.opportunityId
  })
  transport.dispose()
})

class FakeCloudTransport implements CloudCompletionTransport {
  readonly requests: CloudCompletionRequest[] = []
  readonly feedbackEvents: Array<{ completionId: string; feedback: CloudCompletionFeedback }> = []
  onComplete?: (
    request: CloudCompletionRequest,
    options: { signal?: AbortSignal; onDelta?: (text: string) => void }
  ) => Promise<CloudCompletionResult>

  complete(
    request: CloudCompletionRequest,
    options: { signal?: AbortSignal; onDelta?: (text: string) => void }
  ): Promise<CloudCompletionResult> {
    this.requests.push(request)
    return this.onComplete?.(request, options) ?? Promise.reject(new Error('missing fake'))
  }

  feedback(completionId: string, feedback: CloudCompletionFeedback): void {
    this.feedbackEvents.push({ completionId, feedback })
  }
}

function completionInput(prefix: string, suffix = '\n}'): CloudCompletionInput {
  return {
    triggerKind: 'automatic',
    document: { languageId: 'cpp', relativePath: 'src/main.cpp', version: 1 },
    position: { line: 0, character: prefix.length },
    prefix,
    suffix
  }
}

test('client reuses compatible in-flight and cached prefix extensions with exact suffix', async () => {
  const transport = new FakeCloudTransport()
  let resolveRequest!: (result: CloudCompletionResult) => void
  transport.onComplete = (_request, options) => {
    options.onDelta?.('fetch')
    return new Promise(resolve => {
      resolveRequest = resolve
    })
  }
  const client = new CloudInlineCompletionClient(transport, '0.1.2', 'device-session')

  const first = client.complete(completionInput('const x = '))
  const second = client.complete(completionInput('const x = fe'))
  const opportunityId = transport.requests[0]?.opportunityId ?? ''
  resolveRequest({ text: 'fetchData()', completionId: 'cmp_1', opportunityId })

  assert.equal((await first).text, 'fetchData()')
  assert.equal((await second).text, 'tchData()')
  assert.equal((await client.complete(completionInput('const x = fetch'))).text, 'Data()')
  assert.equal(transport.requests.length, 1)

  transport.onComplete = async request => ({
    text: 'other()',
    completionId: 'cmp_2',
    opportunityId: request.opportunityId
  })
  await client.complete(completionInput('const x = fetch', '\n];'))
  assert.equal(transport.requests.length, 2)
  client.dispose()
})

test('402 blocks cloud requests until the UTC reset time', async () => {
  const transport = new FakeCloudTransport()
  let now = Date.parse('2026-08-26T00:00:00Z')
  const resetAt = now + 60_000
  transport.onComplete = async () => {
    throw new CloudCompletionError(
      402,
      'CODE_COMPLETION_QUOTA_EXHAUSTED',
      'quota exhausted',
      undefined,
      resetAt
    )
  }
  const client = new CloudInlineCompletionClient(
    transport,
    '0.1.2',
    'device-session',
    () => now
  )

  await assert.rejects(client.complete(completionInput('a')), CloudCompletionError)
  await assert.rejects(client.complete(completionInput('ab')), (error: unknown) => {
    return error instanceof CloudCompletionError && error.code === 'CODE_COMPLETION_COOLDOWN'
  })
  assert.equal(transport.requests.length, 1)

  now = resetAt
  transport.onComplete = async request => ({
    text: 'c',
    completionId: 'cmp_2',
    opportunityId: request.opportunityId
  })
  assert.equal((await client.complete(completionInput('ab'))).text, 'c')
  assert.equal(transport.requests.length, 2)
  client.dispose()
})

test('provider cancellation aborts the upstream request when no compatible consumer remains', async () => {
  const transport = new FakeCloudTransport()
  let upstreamSignal: AbortSignal | undefined
  transport.onComplete = async (_request, options) => {
    upstreamSignal = options.signal
    await new Promise<void>((_resolve, reject) => {
      options.signal?.addEventListener(
        'abort',
        () => reject(options.signal?.reason ?? new DOMException('Aborted', 'AbortError')),
        { once: true }
      )
    })
    throw new Error('unreachable')
  }
  const client = new CloudInlineCompletionClient(transport, '0.1.2', 'device-session')
  const controller = new AbortController()
  const completion = client.complete(completionInput('const x = '), controller.signal)

  controller.abort()
  await assert.rejects(completion, { name: 'AbortError' })
  await new Promise(resolve => setTimeout(resolve, 5))

  assert.equal(upstreamSignal?.aborted, true)
  client.dispose()
})
