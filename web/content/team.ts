// Team page content. Fill in the TODO fields before publishing.

export const LINKS = {
  repo: "https://github.com/trumen-com/road-vision",
  weights: "https://github.com/trumen-com/road-vision/tree/main/weights",
  predictions: "https://github.com/trumen-com/road-vision/blob/main/predictions_samples.json",
};

export interface Member {
  name: string;
  role: string;
  did: string[];
  github?: string;
  linkedin?: string;
  portfolio?: string;
  proud?: { title: string; text: string; url?: string }[];
}

export const TEAM: Member[] = [
  {
    name: "Asilbek Shodmonov",
    role: "Website, live demo & product",
    did: ["Next.js website, interactive timelines and dashboards", "Live demo backend (FastAPI) and deployment", "Scene map and annotation tooling"],
    github: "https://github.com/TODO",
    linkedin: "https://www.linkedin.com/in/TODO",
    portfolio: "",
    proud: [
      { title: "ExpoUz", text: "Telegram Mini App for sports matchmaking in Tashkent (NestJS, Prisma, PostgreSQL)." },
      { title: "Jarvis", text: "Personal AI assistant: NestJS backend, Telegram bot and React Native app." },
    ],
  },
  {
    name: "Komron Akmalov",
    role: "TODO role",
    did: ["TODO"],
    github: "",
    linkedin: "",
  },
  {
    name: "TODO third member",
    role: "TODO role",
    did: ["TODO"],
    github: "",
    linkedin: "",
  },
];
