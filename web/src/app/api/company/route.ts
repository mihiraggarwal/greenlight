import fs from 'fs';
import { NextRequest, NextResponse } from "next/server";

export const POST = async (req: NextRequest) => {
    const body = await req.json();
    const company = body.company.toLowerCase();
    const json = fs.readFileSync(`../data/${company}/final/master.json`, "utf-8");

    let data = []
    if (json != "") { data = JSON.parse(json); }

    const greenwashing_scores: number[] = []

    fs.readdirSync('../data/').forEach(file => {
        if (file != company) {
            const js = fs.readFileSync(`../data/${file}/final/master.json`, "utf-8");
            if (js != "") {
                const parsed = JSON.parse(js);
                if (parsed.greenwashing_score) {
                    greenwashing_scores.push(parsed.greenwashing_score.score);
                }
            }
        }
    });
    
    let rank = 1;
    for (let i = 0; i < greenwashing_scores.length; i++) {
        if (greenwashing_scores[i] < data.greenwashing_score.score) {
            rank++;
        }
    }
    
    return NextResponse.json({data: data, rank: rank}, {status: 200});
};