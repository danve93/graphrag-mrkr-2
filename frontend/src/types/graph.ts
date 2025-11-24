export interface GraphDocumentRef {
  document_id?: string
  document_name?: string
}

export interface GraphTextUnit {
  id: string
  document_id?: string
  document_name?: string
}

export interface GraphNode {
  id: string
  label: string
  type?: string | null
  community_id?: number | null
  level?: number | null
  degree?: number
  documents?: GraphDocumentRef[]
}

export interface GraphEdge {
  source: string
  target: string
  type?: string | null
  weight?: number
  description?: string | null
  text_units?: GraphTextUnit[]
}

export interface GraphCommunity {
  community_id: number
  level?: number | null
}

export interface GraphResponse {
  nodes: GraphNode[]
  edges: GraphEdge[]
  communities: GraphCommunity[]
  node_types: string[]
}
