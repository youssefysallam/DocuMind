export interface ChatMessage {
  role: "user" | "assistant";
  content: string;
}

export interface RetrievedSource {
  source: string;
  section_title?: string;
  source_type: string;
  score: number;
  layer: string;
  chunk_text?: string;
}

export interface RetrievalLog {
  intent?: string;
  strategy?: string;
  resolved_query?: string;
  original_query?: string;
  sub_queries?: string[];
  hyde_used?: boolean;
  final_raw?: number;
  final_qa?: number;
  qa_neg_used?: boolean;
  multihop?: {
    triggered: boolean;
    supplements_added: number;
    missing_entities: string[];
  };
}

export interface ConsistencyResult {
  is_consistent: boolean;
  confidence: number;
  unsupported_claims?: string[];
  explanation?: string;
}

export interface ChatResponse {
  answer: string;
  retrieved: RetrievedSource[];
  retrieval_log: RetrievalLog;
  consistency: ConsistencyResult | null;
}

export interface EvalData {
  comparison?: {
    rag_v1_same_dataset_95?: Record<string, unknown>;
    rag_v2?: Record<string, unknown>;
  };
  multiturn?: {
    overall?: Record<string, unknown>;
    coreference?: Record<string, unknown>;
    by_turn_position?: Record<string, unknown>;
  };
}

export interface SystemInfo {
  version: string;
  llm_model: string;
  embed_model: string;
  sparse_model: string;
  reranker: string;
  top_k: { dense: number; sparse: number; final: number };
  features: string[];
}
