// Neo4j Schema Initialization for Graph RAG POC
// Usage: cypher-shell -u neo4j -p pocpassword < init_neo4j_schema.cypher

// ============================================
// CONSTRAINTS (Uniqueness)
// ============================================

CREATE CONSTRAINT book_id IF NOT EXISTS
FOR (b:Book) REQUIRE b.id IS UNIQUE;

CREATE CONSTRAINT chapter_id IF NOT EXISTS
FOR (c:Chapter) REQUIRE c.id IS UNIQUE;

CREATE CONSTRAINT taxonomy_id IF NOT EXISTS
FOR (t:Taxonomy) REQUIRE t.id IS UNIQUE;

CREATE CONSTRAINT tier_id IF NOT EXISTS
FOR (t:Tier) REQUIRE t.id IS UNIQUE;

CREATE CONSTRAINT concept_name IF NOT EXISTS
FOR (c:Concept) REQUIRE c.name IS UNIQUE;

// ============================================
// INDEXES (Performance)
// ============================================

CREATE INDEX book_title IF NOT EXISTS
FOR (b:Book) ON (b.title);

CREATE INDEX book_tier IF NOT EXISTS
FOR (b:Book) ON (b.tier);

CREATE INDEX chapter_number IF NOT EXISTS
FOR (c:Chapter) ON (c.number);

CREATE INDEX chapter_book_id IF NOT EXISTS
FOR (c:Chapter) ON (c.book_id);

// Full-text index for keyword search
CREATE FULLTEXT INDEX chapter_keywords IF NOT EXISTS
FOR (c:Chapter) ON EACH [c.title, c.summary];

// ============================================
// TIER NODES (Static Reference Data)
// ============================================

MERGE (t1:Tier {id: "tier-1", level: 1, name: "Language & Syntax", description: "Programming fundamentals"})
MERGE (t2:Tier {id: "tier-2", level: 2, name: "Code Quality", description: "Clean code practices"})
MERGE (t3:Tier {id: "tier-3", level: 3, name: "Architecture", description: "System design patterns"})
MERGE (t4:Tier {id: "tier-4", level: 4, name: "Operations", description: "DevOps and infrastructure"})
MERGE (t5:Tier {id: "tier-5", level: 5, name: "AI/ML", description: "Machine learning and AI"});

// ============================================
// TAXONOMY NODE
// ============================================

MERGE (tx:Taxonomy {
    id: "software-engineering",
    name: "Software Engineering",
    description: "Comprehensive software engineering knowledge base"
});

// ============================================
// VERIFICATION QUERY
// ============================================

// Run this to verify schema creation:
// CALL db.schema.visualization();

RETURN "Schema initialization complete" AS status;
