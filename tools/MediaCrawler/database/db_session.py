# -*- coding: utf-8 -*-
# Copyright (c) 2025 relakkes@gmail.com
#
# This file is part of MediaCrawler project.
# Repository: https://github.com/NanmiCoder/MediaCrawler/blob/main/database/db_session.py
# GitHub: https://github.com/NanmiCoder
# Licensed under NON-COMMERCIAL LEARNING LICENSE 1.1
#
# 声明：本代码仅供学习和研究目的使用。使用者应遵守以下原则：
# 1. 不得用于任何商业用途。
# 2. 使用时应遵守目标平台的使用条款和robots.txt规则。
# 3. 不得进行大规模爬取或对平台造成运营干扰。
# 4. 应合理控制请求频率，避免给目标平台带来不必要的负担。
# 5. 不得用于任何非法或不当的用途。
#
# 详细许可条款请参阅项目根目录下的LICENSE文件。
# 使用本代码即表示您同意遵守上述原则和LICENSE中的所有条款。

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
from contextlib import asynccontextmanager
from .models import Base
import os
import config
from config.db_config import mysql_db_config, sqlite_db_config, postgres_db_config

# Keep a cache of engines
_engines = {}


async def create_database_if_not_exists(db_type: str):
    if db_type == "mysql" or db_type == "db":
        # Connect to the server without a database
        server_url = f"mysql+asyncmy://{mysql_db_config['user']}:{mysql_db_config['password']}@{mysql_db_config['host']}:{mysql_db_config['port']}"
        engine = create_async_engine(server_url, echo=False)
        async with engine.connect() as conn:
            await conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {mysql_db_config['db_name']}"))
        await engine.dispose()
    elif db_type == "postgres":
        # Connect to the default 'postgres' database
        server_url = f"postgresql+asyncpg://{postgres_db_config['user']}:{postgres_db_config['password']}@{postgres_db_config['host']}:{postgres_db_config['port']}/postgres"
        print(f"[init_db] Connecting to Postgres: host={postgres_db_config['host']}, port={postgres_db_config['port']}, user={postgres_db_config['user']}, dbname=postgres")
        # Isolation level AUTOCOMMIT is required for CREATE DATABASE
        engine = create_async_engine(server_url, echo=False, isolation_level="AUTOCOMMIT")
        async with engine.connect() as conn:
            # Check if database exists
            result = await conn.execute(text(f"SELECT 1 FROM pg_database WHERE datname = '{postgres_db_config['db_name']}'"))
            if not result.scalar():
                await conn.execute(text(f"CREATE DATABASE {postgres_db_config['db_name']}"))
        await engine.dispose()


def get_async_engine(db_type: str = None):
    if db_type is None:
        db_type = config.SAVE_DATA_OPTION

    if db_type in ["json", "jsonl", "csv"]:
        return None

    if db_type == "sqlite":
        sqlite_db_path = sqlite_db_config["db_path"]
        if config.SAVE_DATA_PATH:
            platform_db_name = f"{config.PLATFORM}.db" if config.PLATFORM else "sqlite_tables.db"
            sqlite_db_path = os.path.join(config.SAVE_DATA_PATH, platform_db_name)
        sqlite_db_path = os.path.abspath(sqlite_db_path)
        os.makedirs(os.path.dirname(sqlite_db_path), exist_ok=True)
        db_url = f"sqlite+aiosqlite:///{sqlite_db_path}"
        engine_cache_key = f"sqlite::{sqlite_db_path}"
    elif db_type == "mysql" or db_type == "db":
        db_url = f"mysql+asyncmy://{mysql_db_config['user']}:{mysql_db_config['password']}@{mysql_db_config['host']}:{mysql_db_config['port']}/{mysql_db_config['db_name']}"
        engine_cache_key = db_type
    elif db_type == "postgres":
        db_url = f"postgresql+asyncpg://{postgres_db_config['user']}:{postgres_db_config['password']}@{postgres_db_config['host']}:{postgres_db_config['port']}/{postgres_db_config['db_name']}"
        engine_cache_key = db_type
    else:
        raise ValueError(f"Unsupported database type: {db_type}")

    if engine_cache_key in _engines:
        return _engines[engine_cache_key]

    engine = create_async_engine(db_url, echo=False)
    _engines[engine_cache_key] = engine
    return engine


def _get_platform_table_names(platform: str) -> list[str]:
    table_names_by_platform = {
        "bili": [
            "bilibili_video",
            "bilibili_video_comment",
            "bilibili_up_info",
            "bilibili_contact_info",
            "bilibili_up_dynamic",
        ],
        "dy": [
            "douyin_aweme",
            "douyin_aweme_comment",
            "dy_creator",
        ],
        "ks": [
            "kuaishou_video",
            "kuaishou_video_comment",
        ],
        "wb": [
            "weibo_note",
            "weibo_note_comment",
            "weibo_creator",
        ],
        "xhs": [
            "xhs_creator",
            "xhs_note",
            "xhs_note_comment",
        ],
        "tieba": [
            "tieba_note",
            "tieba_comment",
            "tieba_creator",
        ],
        "zhihu": [
            "zhihu_content",
            "zhihu_comment",
            "zhihu_creator",
        ],
    }
    return table_names_by_platform.get(platform, list(Base.metadata.tables.keys()))


async def create_tables(db_type: str = None):
    if db_type is None:
        db_type = config.SAVE_DATA_OPTION
    await create_database_if_not_exists(db_type)
    engine = get_async_engine(db_type)
    if engine:
        async with engine.begin() as conn:
            if db_type == "sqlite":
                table_names = _get_platform_table_names(config.PLATFORM)
                tables = [Base.metadata.tables[name] for name in table_names if name in Base.metadata.tables]
                await conn.run_sync(lambda sync_conn: Base.metadata.create_all(sync_conn, tables=tables))
            else:
                await conn.run_sync(Base.metadata.create_all)


@asynccontextmanager
async def get_session() -> AsyncSession:
    engine = get_async_engine(config.SAVE_DATA_OPTION)
    if not engine:
        yield None
        return
    AsyncSessionFactory = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    session = AsyncSessionFactory()
    try:
        yield session
        await session.commit()
    except Exception as e:
        await session.rollback()
        raise e
    finally:
        await session.close()
