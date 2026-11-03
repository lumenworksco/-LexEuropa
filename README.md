# LexEuropa

### A Custom AI Legal Intelligence System for European Union Law

LexEuropa is an AI legal intelligence project focused on understanding, researching, and reasoning over European Union law.

The goal is to build a specialized legal AI that can assist with legal research, case-law analysis, legislation, document analysis, legal reasoning, and drafting while grounding its answers in authoritative European legal sources.

Rather than relying solely on a general-purpose chatbot, LexEuropa will combine a custom-trained legal language model with a structured European legal knowledge base, retrieval-augmented generation (RAG), multilingual capabilities, and automated citation verification.

The long-term objective is to develop a custom European legal foundation model capable of understanding the complex relationships between EU legislation, regulations, directives, treaties, CJEU case law, and national implementation.

---

## AI Architecture

LexEuropa will use a hybrid architecture rather than attempting to train a large language model completely from scratch at the beginning.

### 1. Foundation Model

The initial system will use a capable open-weight language model as its foundation, with candidate model families including Qwen, Llama, Mistral, and Gemma.

The model will be evaluated specifically for:

- Legal reasoning
- Multilingual understanding
- Long-context processing
- Instruction following
- Citation behavior
- Structured output
- European legal terminology

The selected model will then be adapted to European legal tasks through supervised fine-tuning and parameter-efficient methods such as LoRA or QLoRA.

As the project matures, LexEuropa can progress toward continued pretraining and eventually a fully custom European legal foundation model.

---

## 2. European Legal Knowledge Base

The system will maintain a structured legal corpus containing authoritative European legal materials, including:

- EU treaties
- Regulations
- Directives
- Decisions
- Consolidated legislation
- CJEU judgments
- Advocate General opinions
- Relevant preparatory documents
- Legal metadata and relationships between documents

Where possible, legal documents will retain identifiers such as CELEX numbers, dates, provisions, jurisdictions, subjects, and relationships to other legal authorities.

The objective is to make every important piece of information traceable to its underlying legal source.

---

## 3. Retrieval-Augmented Generation

Retrieval-Augmented Generation will be a core component of LexEuropa.

Instead of relying exclusively on the model's internal knowledge, a user's question will be analyzed and used to retrieve the most relevant legal authorities.

A typical request will follow this process:

**User Question → Legal Query Analysis → Legal Retrieval → Reranking → Model Reasoning → Citation Verification → Grounded Answer**

The retrieval system will combine semantic vector search with traditional keyword/BM25 search.

This allows LexEuropa to find both conceptually similar material and exact legal provisions, articles, case numbers, and terminology.

---

## 4. Legal Reasoning

The model will be trained to perform legal tasks such as:

- Identifying relevant legal issues
- Finding applicable provisions
- Interpreting definitions
- Identifying exceptions and limitations
- Connecting legislation with case law
- Comparing competing legal arguments
- Distinguishing rules from interpretations
- Identifying uncertainty
- Determining when available evidence is insufficient
- Producing structured legal analysis

LexEuropa should not simply produce answers that sound legally convincing. It should be designed to reason from identifiable legal authorities.

---

## 5. Citation and Evidence Verification

A dedicated verification layer will be used to reduce unsupported legal claims.

When the language model produces an answer, the verification system will evaluate whether the cited legal authority actually supports the associated claim.

For example:

**Model claim**

> A Member State may restrict X under a particular provision.

**Verification**

- Is the cited provision relevant?
- Does the provision actually support the claim?
- Has subsequent CJEU case law affected its interpretation?
- Is the source current?
- Is the claim fully supported, partially supported, or unsupported?

The system should be able to identify:

**Supported claims**  
Claims directly supported by authoritative sources.

**Interpretative claims**  
Claims based on judicial interpretation or legal analysis.

**Model analysis**  
Reasoned conclusions derived from the available authorities.

**Uncertain claims**  
Claims where the available evidence is insufficient.

This evidence-first approach is a fundamental part of LexEuropa.

---

## 6. Multilingual European Legal AI

European law is inherently multilingual.

LexEuropa will therefore be designed as a multilingual legal system from the beginning rather than translating everything into a single language.

The system will support legal research across multiple official EU languages while preserving the original authoritative legal texts.

For example, a user could ask a question in Dutch while the retrieval system identifies relevant material in Dutch, English, French, German, or another relevant EU language.

The system will maintain relationships between multilingual versions of the same legal instrument wherever possible.

This is intended to allow LexEuropa to understand that different language versions represent the same underlying European legal instrument while still respecting the importance of the original authoritative wording.

---

## 7. Legal Knowledge Graph

A future component of LexEuropa will be a legal knowledge graph connecting:

**Treaties → Articles → Regulations → Directives → National Implementation → CJEU Cases → Interpretations**

This will allow the system to understand relationships between legal authorities rather than treating every document as an isolated piece of text.

For example, the system could identify that a particular CJEU judgment interprets a specific article of a directive and that the directive has subsequently been amended.

This temporal and relational understanding is particularly important for legal research.

---

## 8. Training Strategy

LexEuropa will follow an incremental model-development strategy.

### Stage 1: Legal RAG

Start with an open-weight foundation model and a high-quality European legal retrieval system.

### Stage 2: Legal Fine-Tuning

Fine-tune the model using carefully constructed examples covering:

- Legal question answering
- Legal reasoning
- Case-law analysis
- Statutory interpretation
- Legal summarization
- Citation generation
- Evidence identification
- Legal document analysis
- Appropriate refusal and uncertainty behavior

### Stage 3: Continued Legal Pretraining

Once a sufficiently large and high-quality legal corpus has been established, investigate continued pretraining on European legal text.

### Stage 4: Custom Foundation Model

Long term, develop a dedicated European legal foundation model trained specifically for multilingual European legal reasoning.

The goal is not simply to create a chatbot that knows legal terminology, but a model that has been optimized around the structure and requirements of European legal research.

---

## 9. Evaluation

LexEuropa will use a dedicated European legal evaluation benchmark rather than relying solely on generic AI benchmarks.

The benchmark will contain carefully constructed legal questions across areas such as:

- GDPR and data protection
- Competition law
- Consumer protection
- Free movement
- Employment law
- Environmental law
- Digital regulation
- AI regulation
- Fundamental rights
- Internal market law
- Tax and VAT
- Corporate law
- CJEU case law

The system will be evaluated on:

- Answer accuracy
- Legal reasoning quality
- Citation accuracy
- Citation completeness
- Source relevance
- Hallucination rate
- Ability to identify uncertainty
- Multilingual consistency
- Temporal accuracy
- Case-law interpretation
- Ability to distinguish law from analysis

Citation and evidence accuracy will be treated as first-class metrics.

---

## 10. Core User Capabilities

LexEuropa will ultimately provide capabilities including:

- EU legal research
- Legal question answering
- CJEU case-law research
- Legislation and regulation analysis
- Contract analysis
- Legal document summarization
- Issue spotting
- Legal argument analysis
- Citation retrieval
- Source verification
- Comparison of legal provisions
- Multilingual legal research
- EU and national law comparison
- Legal drafting assistance

Users should be able to ask a legal question and receive a structured response containing the relevant legal authorities, explanations, citations, and clearly identified areas of uncertainty.

---

## Long-Term Vision

LexEuropa aims to become a specialized European legal intelligence platform built around authoritative sources, transparent reasoning, multilingual legal understanding, and verifiable citations.

The project will begin as a retrieval-augmented legal AI built on an existing open-weight foundation model, progressively adding legal fine-tuning, evidence verification, multilingual capabilities, and a legal knowledge graph.

Over time, the project will evolve toward a custom European legal foundation model specifically optimized for understanding and reasoning over EU law.

**LexEuropa is intended to assist with legal research, analysis, and drafting. It is not intended to replace qualified legal professionals or constitute legal advice.**

---

## Current Status: Stage 1 (Legal RAG)

A working vertical slice of Stage 1 is implemented in `lexeuropa/`, **English-only** (the multilingual vision in §6 above is not yet built), with generation and embeddings served via Hugging Face Inference Providers rather than run locally. See `CLAUDE.md` for setup, commands, exactly what runs where, and known limitations.
