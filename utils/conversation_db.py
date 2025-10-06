#!/usr/bin/env python3
"""
Conversation Database Utilities
================================
Helper functions for managing conversations and messages in Supabase PostgreSQL.
Uses the existing asyncpg connection pool from api_server.
"""

import logging
from typing import List, Dict, Any, Optional
from uuid import UUID
import asyncpg

logger = logging.getLogger(__name__)


async def create_conversation(
    pool: asyncpg.Pool,
    user_id: str,
    title: str
) -> Optional[Dict[str, Any]]:
    """
    Create a new conversation.

    Args:
        pool: Database connection pool
        user_id: User ID (can be session ID for now, UUID for auth later)
        title: Conversation title

    Returns:
        Created conversation dict or None if failed
    """
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO conversations (user_id, title)
                VALUES ($1::uuid, $2)
                RETURNING id, user_id, title, created_at, updated_at
                """,
                user_id,
                title
            )

            if row:
                return {
                    "id": str(row["id"]),
                    "user_id": str(row["user_id"]),
                    "title": row["title"],
                    "created_at": row["created_at"].isoformat(),
                    "updated_at": row["updated_at"].isoformat()
                }
    except Exception as e:
        logger.error(f"Failed to create conversation: {e}", exc_info=True)
        return None


async def get_conversations(
    pool: asyncpg.Pool,
    user_id: str,
    limit: int = 50
) -> List[Dict[str, Any]]:
    """
    Get all conversations for a user.

    Args:
        pool: Database connection pool
        user_id: User ID
        limit: Maximum number of conversations to return

    Returns:
        List of conversation dicts
    """
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT
                    c.id,
                    c.user_id,
                    c.title,
                    c.created_at,
                    c.updated_at,
                    (
                        SELECT m.content
                        FROM messages m
                        WHERE m.conversation_id = c.id
                        ORDER BY m.created_at DESC
                        LIMIT 1
                    ) as last_message
                FROM conversations c
                WHERE c.user_id = $1::uuid
                ORDER BY c.updated_at DESC
                LIMIT $2
                """,
                user_id,
                limit
            )

            return [
                {
                    "id": str(row["id"]),
                    "user_id": str(row["user_id"]),
                    "title": row["title"],
                    "last_message": row["last_message"],
                    "created_at": row["created_at"].isoformat(),
                    "updated_at": row["updated_at"].isoformat()
                }
                for row in rows
            ]
    except Exception as e:
        logger.error(f"Failed to get conversations: {e}", exc_info=True)
        return []


async def get_conversation_messages(
    pool: asyncpg.Pool,
    conversation_id: str
) -> List[Dict[str, Any]]:
    """
    Get all messages for a conversation.

    Args:
        pool: Database connection pool
        conversation_id: Conversation ID

    Returns:
        List of message dicts
    """
    try:
        async with pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT id, conversation_id, role, content, created_at
                FROM messages
                WHERE conversation_id = $1::uuid
                ORDER BY created_at ASC
                """,
                conversation_id
            )

            return [
                {
                    "id": str(row["id"]),
                    "conversation_id": str(row["conversation_id"]),
                    "role": row["role"],
                    "content": row["content"],
                    "created_at": row["created_at"].isoformat()
                }
                for row in rows
            ]
    except Exception as e:
        logger.error(f"Failed to get messages: {e}", exc_info=True)
        return []


async def save_message(
    pool: asyncpg.Pool,
    conversation_id: str,
    role: str,
    content: str
) -> Optional[Dict[str, Any]]:
    """
    Save a message to a conversation.

    Args:
        pool: Database connection pool
        conversation_id: Conversation ID
        role: Message role ('user' or 'assistant')
        content: Message content

    Returns:
        Created message dict or None if failed
    """
    try:
        async with pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                INSERT INTO messages (conversation_id, role, content)
                VALUES ($1::uuid, $2, $3)
                RETURNING id, conversation_id, role, content, created_at
                """,
                conversation_id,
                role,
                content
            )

            if row:
                return {
                    "id": str(row["id"]),
                    "conversation_id": str(row["conversation_id"]),
                    "role": row["role"],
                    "content": row["content"],
                    "created_at": row["created_at"].isoformat()
                }
    except Exception as e:
        logger.error(f"Failed to save message: {e}", exc_info=True)
        return None


async def delete_conversation(
    pool: asyncpg.Pool,
    conversation_id: str,
    user_id: str
) -> bool:
    """
    Delete a conversation and all its messages.

    Args:
        pool: Database connection pool
        conversation_id: Conversation ID
        user_id: User ID (for verification)

    Returns:
        True if deleted, False otherwise
    """
    try:
        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM conversations
                WHERE id = $1::uuid AND user_id = $2::uuid
                """,
                conversation_id,
                user_id
            )

            # Check if any rows were deleted
            return result.split()[-1] != "0"
    except Exception as e:
        logger.error(f"Failed to delete conversation: {e}", exc_info=True)
        return False


async def update_conversation_title(
    pool: asyncpg.Pool,
    conversation_id: str,
    user_id: str,
    title: str
) -> bool:
    """
    Update a conversation title.

    Args:
        pool: Database connection pool
        conversation_id: Conversation ID
        user_id: User ID (for verification)
        title: New title

    Returns:
        True if updated, False otherwise
    """
    try:
        async with pool.acquire() as conn:
            result = await conn.execute(
                """
                UPDATE conversations
                SET title = $3, updated_at = NOW()
                WHERE id = $1::uuid AND user_id = $2::uuid
                """,
                conversation_id,
                user_id,
                title
            )

            return result.split()[-1] != "0"
    except Exception as e:
        logger.error(f"Failed to update conversation title: {e}", exc_info=True)
        return False
