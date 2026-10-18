const fs = require("fs");
const path = require("path");
const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, AlignmentType,
  Table, TableRow, TableCell, WidthType, ShadingType, BorderStyle,
  ImageRun, PageBreak, TableOfContents, LevelFormat, convertInchesToTwip,
  VerticalAlign, PageNumber, Footer, Header
} = require("docx");

const OUT = path.join(__dirname, "outputs");
const results = JSON.parse(fs.readFileSync(path.join(OUT, "results.json"), "utf-8"));

// ---------- helpers ----------
const FONT = "Calibri";
const COLOR_PRIMARY = "1F4E79";
const COLOR_ACCENT = "2E74B5";

function h1(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_1,
    spacing: { before: 320, after: 160 },
    children: [new TextRun({ text, bold: true, color: COLOR_PRIMARY, font: FONT })],
  });
}
function h2(text) {
  return new Paragraph({
    heading: HeadingLevel.HEADING_2,
    spacing: { before: 240, after: 120 },
    children: [new TextRun({ text, bold: true, color: COLOR_ACCENT, font: FONT })],
  });
}
function p(text, opts = {}) {
  return new Paragraph({
    spacing: { after: 160, line: 288 },
    children: [new TextRun({ text, font: FONT, size: 22, ...opts })],
  });
}
function bullet(text) {
  return new Paragraph({
    numbering: { reference: "bullet-list", level: 0 },
    spacing: { after: 80 },
    children: [new TextRun({ text, font: FONT, size: 22 })],
  });
}
function caption(text) {
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { after: 240 },
    children: [new TextRun({ text, italics: true, font: FONT, size: 20, color: "555555" })],
  });
}
function imageParagraph(fileName, widthPx, heightPx, maxWidthEmuIn = 5.8) {
  const buf = fs.readFileSync(path.join(OUT, fileName));
  const ratio = heightPx / widthPx;
  const widthIn = maxWidthEmuIn;
  const heightIn = widthIn * ratio;
  return new Paragraph({
    alignment: AlignmentType.CENTER,
    spacing: { before: 120, after: 80 },
    children: [
      new ImageRun({
        data: buf,
        type: "png",
        transformation: {
          width: Math.round(widthIn * 96),
          height: Math.round(heightIn * 96),
        },
      }),
    ],
  });
}
function cell(text, opts = {}) {
  const { bold = false, shade = null, width = null, align = AlignmentType.LEFT } = opts;
  return new TableCell({
    width: width ? { size: width, type: WidthType.DXA } : undefined,
    shading: shade ? { type: ShadingType.CLEAR, fill: shade } : undefined,
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 80, bottom: 80, left: 100, right: 100 },
    children: [new Paragraph({
      alignment: align,
      children: [new TextRun({ text: String(text), bold, font: FONT, size: 20 })],
    })],
  });
}
function fmtNum(n, digits = 4) {
  return Number(n).toFixed(digits);
}
function fmtUSD(n) {
  return "$" + Number(n).toLocaleString("en-US", { maximumFractionDigits: 0 });
}

// ---------- Cover page ----------
const coverChildren = [
  new Paragraph({ spacing: { after: 100 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "TRƯỜNG ĐẠI HỌC", bold: true, size: 26, font: FONT })] }),
  new Paragraph({ spacing: { after: 100 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "KHOA CÔNG NGHỆ THÔNG TIN", bold: true, size: 26, font: FONT })] }),
  new Paragraph({ spacing: { after: 600 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "─────────────────", size: 26, font: FONT })] }),

  new Paragraph({ spacing: { before: 800, after: 200 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "BÁO CÁO BÀI TẬP LỚN / LAB 02", bold: true, size: 32, color: COLOR_PRIMARY, font: FONT })] }),
  new Paragraph({ spacing: { after: 100 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "HOUSE PRICES:", bold: true, size: 40, color: COLOR_PRIMARY, font: FONT })] }),
  new Paragraph({ spacing: { after: 600 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "ADVANCED REGRESSION TECHNIQUES", bold: true, size: 40, color: COLOR_PRIMARY, font: FONT })] }),

  new Paragraph({ spacing: { after: 100 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Dự đoán giá bán nhà ở Ames, Iowa bằng các kỹ thuật hồi quy nâng cao", italics: true, size: 24, font: FONT })] }),

  new Paragraph({ spacing: { before: 900, after: 120 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Môn học: Khai phá dữ liệu / Học máy", size: 24, font: FONT })] }),

  new Paragraph({ spacing: { before: 700, after: 120 }, alignment: AlignmentType.CENTER,
    children: [new TextRun({ text: "Danh sách thành viên nhóm", bold: true, size: 24, font: FONT })] }),
];

const coverPage = [
  ...coverChildren,
];

// ---------- Section 0: table header text white fix via re-creating cells ----------
function headerCell(text, width) {
  return new TableCell({
    width: { size: width, type: WidthType.DXA },
    shading: { type: ShadingType.CLEAR, fill: "1F4E79" },
    verticalAlign: VerticalAlign.CENTER,
    margins: { top: 80, bottom: 80, left: 100, right: 100 },
    children: [new Paragraph({
      alignment: AlignmentType.CENTER,
      children: [new TextRun({ text, bold: true, color: "FFFFFF", font: FONT, size: 20 })],
    })],
  });
}

const memberTableFixed = new Table({
  width: { size: 7000, type: WidthType.DXA },
  alignment: AlignmentType.CENTER,
  rows: [
    new TableRow({ tableHeader: true, children: [headerCell("MSSV", 2000), headerCell("Họ và tên", 5000)] }),
    new TableRow({ children: [cell("3124411174", { width: 2000, align: AlignmentType.CENTER }), cell("Tất Võ Tiến Nam", { width: 5000 })] }),
    new TableRow({ children: [cell("3124411175", { width: 2000, align: AlignmentType.CENTER }), cell("Phạm Anh Minh", { width: 5000 })] }),
  ],
});

// ---------- Section 1: Phân công công việc ----------
const assignmentTable = new Table({
  width: { size: 9000, type: WidthType.DXA },
  rows: [
    new TableRow({ tableHeader: true, children: [
      headerCell("MSSV - Họ tên", 2600),
      headerCell("Công việc phụ trách", 4800),
      headerCell("Tỷ lệ đóng góp", 1600),
    ]}),
    new TableRow({ children: [
      cell("3124411174\nTất Võ Tiến Nam", {}),
      cell("Đọc và phân tích paper Ames Housing của Dean De Cock; thực hiện Data Understanding & Data Preparation (xử lý dữ liệu thiếu, mã hoá one-hot); xây dựng và huấn luyện các mô hình tuyến tính (Linear, Ridge, Lasso). "),
      cell("40%", { align: AlignmentType.CENTER }),
    ]}),
    new TableRow({ children: [
      cell("3124411166\nPhạm Anh Minh", {}),
      cell("Xây dựng và huấn luyện các mô hình ensemble (Random Forest, Gradient Boosting); đánh giá, so sánh mô hình bằng RMSE/MAE/R2 và cross-validation; trực quan hoá kết quả (feature importance, actual vs predicted)."),
      cell("30%", { align: AlignmentType.CENTER }),
    ]}),
    new TableRow({ children: [
      cell("3124411116\nLê Long Hưng", {}),
      cell("Tổng hợp phần Business Understanding và Data Understanding của báo cáo,tổng hợp phần Modeling, Evaluation, Deployment và Kết luận."),
      cell("30%", { align: AlignmentType.CENTER }),
    ]}),
  ],
});

// ---------- Section: missing values table ----------
const missingRows = Object.entries(results.missing_top);
const missingTable = new Table({
  width: { size: 9000, type: WidthType.DXA },
  rows: [
    new TableRow({ tableHeader: true, children: [
      headerCell("Cột dữ liệu", 4000),
      headerCell("% giá trị thiếu", 2500),
      headerCell("Xử lý", 2500),
    ]}),
    ...missingRows.map(([col, pct]) => {
      const isDrop = ["PoolQC", "MiscFeature", "Alley", "Fence"].includes(col);
      return new TableRow({ children: [
        cell(col),
        cell(fmtNum(pct, 2) + "%", { align: AlignmentType.CENTER }),
        cell(isDrop ? "Loại bỏ cột (>70% thiếu)" : "Điền mean/mode", { align: AlignmentType.CENTER }),
      ]});
    }),
  ],
});

// ---------- Section: correlation table ----------
const corrEntries = Object.entries(results.top_corr_with_saleprice).filter(([k]) => k !== "SalePrice");
const corrTable = new Table({
  width: { size: 9000, type: WidthType.DXA },
  rows: [
    new TableRow({ tableHeader: true, children: [
      headerCell("Biến số", 5000),
      headerCell("Hệ số tương quan với SalePrice", 4000),
    ]}),
    ...corrEntries.map(([col, v]) => new TableRow({ children: [
      cell(col),
      cell(fmtNum(v, 3), { align: AlignmentType.CENTER }),
    ]})),
  ],
});

// ---------- Section: model metrics table ----------
const metricEntries = Object.entries(results.model_metrics);
const metricsTable = new Table({
  width: { size: 9500, type: WidthType.DXA },
  rows: [
    new TableRow({ tableHeader: true, children: [
      headerCell("Mô hình", 2100),
      headerCell("RMSE (log)", 1500),
      headerCell("CV RMSE 5-fold (log)", 1900),
      headerCell("RMSE ($)", 1500),
      headerCell("MAE ($)", 1500),
      headerCell("R²", 1000),
    ]}),
    ...metricEntries.map(([name, m]) => {
      const isBest = name === results.best_model;
      return new TableRow({ children: [
        cell(name, { bold: isBest, shade: isBest ? "DDEBF7" : null }),
        cell(fmtNum(m.RMSE_log), { align: AlignmentType.CENTER, bold: isBest, shade: isBest ? "DDEBF7" : null }),
        cell(fmtNum(m.CV_RMSE_log_5fold), { align: AlignmentType.CENTER, bold: isBest, shade: isBest ? "DDEBF7" : null }),
        cell(fmtUSD(m.RMSE_USD), { align: AlignmentType.CENTER, bold: isBest, shade: isBest ? "DDEBF7" : null }),
        cell(fmtUSD(m.MAE_USD), { align: AlignmentType.CENTER, bold: isBest, shade: isBest ? "DDEBF7" : null }),
        cell(fmtNum(m.R2, 3), { align: AlignmentType.CENTER, bold: isBest, shade: isBest ? "DDEBF7" : null }),
      ]});
    }),
  ],
});

// ---------- Section: feature importance table ----------
const fiEntries = Object.entries(results.top_feature_importances).slice(0, 10);
const fiTable = new Table({
  width: { size: 9000, type: WidthType.DXA },
  rows: [
    new TableRow({ tableHeader: true, children: [
      headerCell("Hạng", 800),
      headerCell("Đặc trưng", 4500),
      headerCell("Mức độ quan trọng", 3700),
    ]}),
    ...fiEntries.map(([col, v], i) => new TableRow({ children: [
      cell(String(i + 1), { align: AlignmentType.CENTER }),
      cell(col),
      cell(fmtNum(v, 4), { align: AlignmentType.CENTER }),
    ]})),
  ],
});

// ---------- Build document ----------
const doc = new Document({
  numbering: {
    config: [{
      reference: "bullet-list",
      levels: [{ level: 0, format: LevelFormat.BULLET, text: "•", alignment: AlignmentType.LEFT,
        style: { paragraph: { indent: { left: 480, hanging: 240 } } } }],
    }],
  },
  styles: {
    default: {
      document: { run: { font: FONT, size: 22 } },
    },
  },
  sections: [
    // ---- COVER PAGE (no header/footer) ----
    {
      properties: { page: { size: { width: 12240, height: 15840 },
        margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
      children: [
        ...coverPage,
        memberTableFixed,
        new Paragraph({ spacing: { before: 900 }, alignment: AlignmentType.CENTER,
          children: [new TextRun({ text: "Tháng 9, 2026", size: 22, font: FONT })] }),
        new Paragraph({ children: [new PageBreak()] }),
      ],
    },
    // ---- MAIN CONTENT ----
    {
      properties: { page: { size: { width: 12240, height: 15840 },
        margin: { top: 1440, bottom: 1440, left: 1440, right: 1440 } } },
      headers: {
        default: new Header({ children: [new Paragraph({
          alignment: AlignmentType.RIGHT,
          children: [new TextRun({ text: "Lab 02 - House Prices Advanced Regression", size: 16, color: "888888", font: FONT })],
        })] }),
      },
      footers: {
        default: new Footer({ children: [new Paragraph({
          alignment: AlignmentType.CENTER,
          children: [
            new TextRun({ text: "Trang ", size: 18, font: FONT, color: "888888" }),
            new TextRun({ children: [PageNumber.CURRENT], size: 18, font: FONT, color: "888888" }),
            new TextRun({ text: " / ", size: 18, font: FONT, color: "888888" }),
            new TextRun({ children: [PageNumber.TOTAL_PAGES], size: 18, font: FONT, color: "888888" }),
          ],
        })] }),
      },
      children: [
        h1("MỤC LỤC"),
        p("1. Phân công công việc"),
        p("2. Business Understanding (Hiểu bài toán)"),
        p("3. Data Understanding (Hiểu dữ liệu)"),
        p("4. Data Preparation (Tiền xử lý dữ liệu)"),
        p("5. Modeling (Xây dựng mô hình)"),
        p("6. Evaluation (Đánh giá mô hình)"),
        p("7. Deployment (Triển khai / mô phỏng nộp kết quả)"),
        p("8. Kết luận"),
        p("9. Tài liệu tham khảo"),
        new Paragraph({ children: [new PageBreak()] }),

        // 1. PHÂN CÔNG CÔNG VIỆC
        h1("1. PHÂN CÔNG CÔNG VIỆC"),
        assignmentTable,
        p(""),

        // 2. BUSINESS UNDERSTANDING
        h1("2. BUSINESS UNDERSTANDING (HIỂU BÀI TOÁN)"),
        p("Bài toán được xây dựng dựa trên bộ dữ liệu Ames Housing do Dean De Cock biên soạn cho mục đích giáo dục, được Kaggle sử dụng trong cuộc thi \"House Prices: Advanced Regression Techniques\" như một phiên bản mở rộng và hiện đại hơn so với bộ dữ liệu Boston Housing kinh điển."),
        p("Mục tiêu: dự đoán giá bán cuối cùng (SalePrice) của các căn nhà ở thành phố Ames, bang Iowa, dựa trên 79 biến giải thích mô tả gần như mọi khía cạnh của một căn nhà ở (diện tích, chất lượng vật liệu, năm xây dựng, vị trí khu vực, số phòng, ga-ra, tầng hầm, v.v.)."),
        p("Đây là bài toán hồi quy (regression) giám sát, với thước đo đánh giá được đề xuất trong paper gốc và cuộc thi Kaggle là Root Mean Squared Error (RMSE) tính trên logarit của giá bán, nhằm giảm ảnh hưởng của các căn nhà có giá trị rất cao (outliers) lên hàm mất mát."),
        p("Giá trị thực tiễn: một mô hình dự đoán tốt giúp người mua/bán nhà, ngân hàng thẩm định và các công ty bất động sản ước lượng giá trị hợp lý của căn nhà dựa trên đặc điểm kỹ thuật, thay vì chỉ dựa vào số phòng ngủ hay diện tích lô đất như cách định giá truyền thống."),

        // 3. DATA UNDERSTANDING
        h1("3. DATA UNDERSTANDING (HIỂU DỮ LIỆU)"),
        h2("3.1. Tổng quan dữ liệu"),
        p(`Bộ dữ liệu gồm hai tệp: train.csv với ${results.train_shape[0]} quan sát và ${results.train_shape[1]} cột (bao gồm nhãn SalePrice), và test.csv với ${results.test_shape[0]} quan sát và ${results.test_shape[1]} cột (không có nhãn, dùng để nộp kết quả dự đoán lên Kaggle). Ngoài ra còn có tệp data_description.txt mô tả chi tiết ý nghĩa của từng cột.`),
        p(`Biến mục tiêu SalePrice có giá trị trung bình khoảng ${fmtUSD(results.saleprice_describe.mean)}, độ lệch chuẩn ${fmtUSD(results.saleprice_describe.std)}, dao động từ ${fmtUSD(results.saleprice_describe.min)} đến ${fmtUSD(results.saleprice_describe.max)}. Phân phối của SalePrice bị lệch phải (right-skewed) do một số ít căn nhà có giá trị rất cao — sau khi lấy log(1+SalePrice), phân phối trở nên gần với phân phối chuẩn hơn, thuận lợi hơn cho các mô hình hồi quy tuyến tính.`),
        imageParagraph("target_distribution.png", 1800, 675, 6.2),
        caption("Hình 1. Phân phối SalePrice gốc (trái) và sau khi biến đổi log (phải)"),

        h2("3.2. Phân tích dữ liệu thiếu (missing values)"),
        p("Kiểm tra trên cả train và test cho thấy một số cột có tỉ lệ giá trị thiếu rất cao, đặc biệt là các cột mô tả tiện ích không phải nhà nào cũng có (hồ bơi, hàng rào, lối đi phụ, tiện ích khác):"),
        imageParagraph("missing_values.png", 1350, 900, 5.6),
        caption("Hình 2. Top 20 cột có tỷ lệ dữ liệu thiếu cao nhất (tập train)"),
        missingTable,
        p(""),

        h2("3.3. Phân tích tương quan với biến mục tiêu"),
        p("Trong số các biến số (numeric), 10 biến có tương quan tuyến tính (Pearson) cao nhất với SalePrice được liệt kê dưới đây. Chất lượng tổng thể của căn nhà (OverallQual) và diện tích sàn sống (GrLivArea) là hai yếu tố có ảnh hưởng mạnh nhất."),
        corrTable,
        p(""),
        imageParagraph("correlation_heatmap.png", 1350, 900, 5.6),
        caption("Hình 3. Ma trận tương quan giữa 10 biến số hàng đầu và SalePrice"),

        // 4. DATA PREPARATION
        h1("4. DATA PREPARATION (TIỀN XỬ LÝ DỮ LIỆU)"),
        p("Dựa trên kết quả phân tích dữ liệu thiếu, nhóm thực hiện tiền xử lý theo các bước sau:"),
        bullet("Loại bỏ cột Id vì không mang thông tin dự đoán."),
        bullet("Loại bỏ 4 cột có tỷ lệ thiếu trên 70% ở cả train và test: Alley, PoolQC, Fence, MiscFeature."),
        bullet("Gộp (concat) train và test trước khi xử lý, nhằm đảm bảo các giá trị phân loại (category) chỉ xuất hiện ở một trong hai tập vẫn được mã hoá nhất quán giữa hai tập."),
        bullet(`Với ${results.n_numeric_features} cột số (numeric): điền giá trị thiếu bằng giá trị trung bình (mean) của cột.`),
        bullet(`Với ${results.n_categorical_features} cột phân loại (categorical): điền giá trị thiếu bằng giá trị xuất hiện nhiều nhất (mode) của cột.`),
        bullet(`Mã hoá one-hot encoding (pandas.get_dummies, drop_first=True) cho toàn bộ biến phân loại, thu được ${results.n_features_after_encoding} đặc trưng đầu vào cho mô hình.`),
        bullet("Biến đổi log1p lên biến mục tiêu SalePrice để giảm độ lệch của phân phối, huấn luyện mô hình trên thang log rồi biến đổi ngược (expm1) khi dự đoán."),
        bullet("Chia tập train thành 80% huấn luyện (training) và 20% kiểm định (validation) để đánh giá mô hình một cách khách quan trước khi dự đoán trên tập test thật của Kaggle."),

        // 5. MODELING
        h1("5. MODELING (XÂY DỰNG MÔ HÌNH)"),
        p("Nhóm tiến hành huấn luyện và so sánh 5 mô hình hồi quy khác nhau, từ đơn giản đến phức tạp, nhằm tìm ra mô hình phù hợp nhất với dữ liệu:"),
        bullet("Linear Regression: mô hình hồi quy tuyến tính cơ bản, dùng làm baseline."),
        bullet("Ridge Regression (alpha=10): hồi quy tuyến tính có regularization L2, giúp giảm overfitting khi số lượng đặc trưng lớn (235 đặc trưng sau one-hot)."),
        bullet("Lasso Regression (alpha=0.001): hồi quy tuyến tính có regularization L1, có khả năng đưa hệ số của các đặc trưng ít quan trọng về 0 (feature selection tự động)."),
        bullet("Random Forest Regressor (400 cây): mô hình ensemble dựa trên bagging nhiều cây quyết định, có khả năng nắm bắt quan hệ phi tuyến."),
        bullet("Gradient Boosting Regressor (500 vòng lặp, learning_rate=0.05): mô hình ensemble dựa trên boosting tuần tự, thường cho độ chính xác cao trên dữ liệu dạng bảng (tabular)."),
        p("Tất cả mô hình được huấn luyện trên thang log của SalePrice; hiệu năng được đánh giá bằng RMSE trên thang log (chỉ số dùng để xếp hạng trong cuộc thi Kaggle gốc), cùng với RMSE/MAE quy đổi ra đơn vị USD và hệ số xác định R² để dễ diễn giải, đồng thời thực hiện 5-fold cross-validation trên toàn bộ tập train để kiểm tra tính ổn định của kết quả."),

        // 6. EVALUATION
        h1("6. EVALUATION (ĐÁNH GIÁ MÔ HÌNH)"),
        h2("6.1. So sánh các mô hình"),
        p("Bảng dưới đây tổng hợp kết quả đánh giá trên tập validation (20% dữ liệu train được giữ lại) và kết quả 5-fold cross-validation trên toàn bộ tập train. Mô hình có RMSE (log) thấp nhất được tô nền và in đậm."),
        metricsTable,
        p(""),
        imageParagraph("model_comparison.png", 1200, 750, 5.6),
        caption("Hình 4. So sánh RMSE (thang log) giữa các mô hình trên tập validation"),
        p(`Kết quả cho thấy ${results.best_model} đạt hiệu năng tốt nhất trên tập validation với RMSE (log) = ${fmtNum(results.model_metrics[results.best_model].RMSE_log)}, tương đương sai số trung bình khoảng ${fmtUSD(results.model_metrics[results.best_model].RMSE_USD)} và giải thích được khoảng ${(results.model_metrics[results.best_model].R2 * 100).toFixed(1)}% phương sai của giá nhà (R² = ${fmtNum(results.model_metrics[results.best_model].R2, 3)}). Điều này phù hợp với nhận định phổ biến trong cộng đồng Kaggle rằng với bộ dữ liệu Ames Housing sau khi one-hot encoding tạo ra không gian đặc trưng có số chiều tương đối lớn so với số quan sát, các mô hình tuyến tính có regularization (Ridge/Lasso) thường cho kết quả cạnh tranh, thậm chí vượt trội so với các mô hình cây khi chưa tinh chỉnh siêu tham số kỹ lưỡng.`),

        h2("6.2. Biểu đồ Thực tế và Dự đoán"),
        imageParagraph("actual_vs_predicted.png", 975, 900, 4.6),
        caption(`Hình 5. So sánh giá thực tế và giá dự đoán trên tập validation - mô hình ${results.best_model} (đường đỏ nét đứt là dự đoán hoàn hảo)`),
        p("Các điểm dữ liệu bám sát đường chéo cho thấy mô hình dự đoán khá chính xác ở khoảng giá phổ biến (100.000 - 300.000 USD); sai số có xu hướng tăng nhẹ ở phân khúc nhà giá rất cao do số lượng quan sát ở phân khúc này ít hơn."),

        h2("6.3. Mức độ quan trọng của các đặc trưng (Feature Importance)"),
        p("Sử dụng mô hình Random Forest để xếp hạng mức độ đóng góp của từng đặc trưng vào kết quả dự đoán:"),
        imageParagraph("feature_importance.png", 1350, 1050, 5.6),
        caption("Hình 6. Top 15 đặc trưng quan trọng nhất theo Random Forest"),
        fiTable,
        p(""),
        p("Kết quả phù hợp với phân tích tương quan ở phần Data Understanding: OverallQual (chất lượng tổng thể) và GrLivArea (diện tích sàn sống) là hai yếu tố chi phối mạnh nhất đến giá bán nhà, tiếp theo là các đặc trưng liên quan đến diện tích tầng hầm, ga-ra và năm xây dựng."),

        // 7. DEPLOYMENT
        h1("7. DEPLOYMENT (TRIỂN KHAI / MÔ PHỎNG NỘP KẾT QUẢ)"),
        p(`Sau khi lựa chọn mô hình tốt nhất (${results.best_model}) dựa trên tập validation, nhóm huấn luyện lại mô hình này trên toàn bộ tập train (bao gồm cả phần validation) và sử dụng mô hình đã huấn luyện để dự đoán SalePrice cho 1.459 căn nhà trong tập test.csv của Kaggle.`),
        p("Kết quả dự đoán được xuất ra tệp submission.csv theo đúng định dạng yêu cầu của cuộc thi (hai cột Id và SalePrice), sẵn sàng để nộp lên hệ thống chấm điểm của Kaggle nhằm đánh giá khách quan trên tập test ẩn (private leaderboard)."),
        p("Trong thực tế triển khai sản phẩm, mô hình này có thể được đóng gói thành một dịch vụ (API) nhận đầu vào là các đặc trưng của căn nhà và trả về giá ước tính, phục vụ cho các nền tảng định giá bất động sản trực tuyến."),

        // 8. CONCLUSION
        h1("8. KẾT LUẬN"),
        bullet("Nhóm đã thực hiện đầy đủ quy trình CRISP-DM cho bài toán dự đoán giá nhà: từ hiểu bài toán, khám phá dữ liệu, tiền xử lý, xây dựng nhiều mô hình hồi quy, đánh giá và mô phỏng triển khai."),
        bullet(`Mô hình ${results.best_model} cho kết quả tốt nhất trong số 5 mô hình thử nghiệm, với RMSE (log) ≈ ${fmtNum(results.model_metrics[results.best_model].RMSE_log)} trên tập validation.`),
        bullet("Chất lượng tổng thể (OverallQual) và diện tích sàn sống (GrLivArea) là hai đặc trưng có ảnh hưởng lớn nhất đến giá bán nhà."),
        bullet("Hướng phát triển tiếp theo: xử lý outlier chi tiết hơn, kỹ thuật feature engineering nâng cao (tạo biến tổng diện tích, tuổi nhà tại thời điểm bán...), thử nghiệm các mô hình mạnh hơn như XGBoost/LightGBM và kỹ thuật stacking/blending nhiều mô hình để cải thiện độ chính xác."),

        // 9. REFERENCES
        h1("9. TÀI LIỆU THAM KHẢO"),
        p("[1] De Cock, D. (2011). Ames, Iowa: Alternative to the Boston Housing Data as an End of Semester Regression Project. Journal of Statistics Education, 19(3). https://jse.amstat.org/v19n3/decock.pdf", { italics: true }),
        p("[2] Kaggle. House Prices - Advanced Regression Techniques. https://www.kaggle.com/competitions/house-prices-advanced-regression-techniques", { italics: true }),
        p("[3] Nimje, R. House Prices: Advanced Regression Techniques (notebook tham khảo được cung cấp trong đề bài).", { italics: true }),
        p("[4] Pedregosa, F. et al. (2011). Scikit-learn: Machine Learning in Python. Journal of Machine Learning Research, 12, 2825-2830.", { italics: true }),
      ],
    },
  ],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.writeFileSync(path.join(__dirname, "outputs", "BaoCao_HousePrices_Lab02.docx"), buffer);
  console.log("Report written.");
});
