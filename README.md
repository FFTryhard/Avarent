\#AVARENT API

## What it is

FastAPI wrapper around proprietary C++ market regime engine.

## Run locally

uvicorn app.main:app --reload

## Frontend

npm run dev

## Current endpoints

GET /v1/latest
GET /v1/health

## Current dependency

mga\_module (pybind11 compiled extension)

## Goal

Deploy public API for paying clients.



\#Build C++ modules (windows)



Requires:

\- CMake

\- Ninja

\- Visual Studio Build Tools / MSVC

\- vcpkg



cmake -S . -B build ^

&#x20; -G Ninja ^

&#x20; -DCMAKE\_BUILD\_TYPE=Release ^

&#x20; -DCMAKE\_TOOLCHAIN\_FILE=C:/Users/H/vcpkg/scripts/buildsystems/vcpkg.cmake



cmake --build build



\## C++ Dependencies



\- pybind11

\- Eigen3

\- vcpkg toolchain currently used on Windows



\## Python

* 3.12

