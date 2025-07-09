import fs from 'fs';
import { NextResponse } from "next/server";

export const GET = async () => {

    const data: string[] = []

    fs.readdirSync('../data/').forEach(file => {
        if (fs.existsSync(`../data/${file}/final/master.json`)) {
            data.push(file);
        }
    });
    
    return NextResponse.json({data: data}, {status: 200});
};