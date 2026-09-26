// Team page content.

export const LINKS = {
  repo: "https://github.com/trumen-com/road-vision",
  weights: "https://github.com/trumen-com/road-vision/tree/main/weights",
  predictions: "https://github.com/trumen-com/road-vision/blob/main/predictions_samples.json",
};

export interface Member {
  name: string;
  role: string;
  about?: string;
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
    about: "ERP Project Manager & Business Analyst at Technogym Uzbekistan · BSc Business Information Systems (First Class), WIUT",
    did: ["Next.js website, interactive timelines and dashboards", "Live demo backend (FastAPI) and deployment", "Scene map and annotation tooling"],
    proud: [
      { title: "ExpoUz", text: "Telegram Mini App for sports matchmaking in Tashkent (NestJS, Prisma, PostgreSQL)." },
      { title: "Jarvis", text: "Personal AI assistant: NestJS backend, Telegram bot and React Native app." },
    ],
  },
  {
    name: "Komron Akmalov",
    role: "Computer vision & modelling",
    about: "AI & Computer Vision Engineer at zehnmind.ai · Westminster International University in Tashkent",
    did: ["Detection and tracking pipeline (YOLO11 + ByteTrack-style tracker)", "Event rules, scene alignment and the traffic-light reader", "Causal accident-risk estimator (Part B)"],
    linkedin: "https://www.linkedin.com/in/komron-akmalov-8b6024297/",
  },
  {
    name: "Ashraf Shermatov",
    role: "Project management, data & evaluation",
    about: "Intern Project Manager / Junior Product Manager · BSc Computer Science, Westminster International University in Tashkent",
    did: ["Planning, scope and submission checklist", "Exploratory data analysis of the sample videos", "Labelling the samples, error review and the technical report"],
    linkedin: "https://www.linkedin.com/in/ashraf-shermatov-675465252/",
  },
];
