package com.kimminh.moneysense.ui.history

import android.annotation.SuppressLint
import android.view.LayoutInflater
import android.view.ViewGroup
import androidx.recyclerview.widget.RecyclerView
import com.trung.moneyrecognizer.R
import com.trung.moneyrecognizer.databinding.HistoryItemBinding

class HistoryAdapter : RecyclerView.Adapter<HistoryAdapter.HistoryHolder>() {

    private var historyList = emptyList<HistoryEntity>()
    class HistoryHolder(private val binding: HistoryItemBinding): RecyclerView.ViewHolder(binding.root) {
        fun bind(history: HistoryEntity) {
            val total = "${history.totalMoney} VND"

            binding.tvDateCreated.text = history.date
            binding.tvTotalMoney.text = total
            binding.tvMoneyTypes.text = binding.root.resources.getString(R.string.money_types, history.moneyTypes)
        }
    }

    override fun onCreateViewHolder(parent: ViewGroup, viewType: Int): HistoryHolder {
        val binding = HistoryItemBinding.inflate(LayoutInflater.from(parent.context), parent, false)
        return HistoryHolder(binding)
    }

    override fun onBindViewHolder(historyHolder: HistoryHolder, position: Int) {
        val history: HistoryEntity = historyList[position]
        historyHolder.bind(history)

    }

    override fun getItemCount(): Int {
        return historyList.size
    }

    @SuppressLint("NotifyDataSetChanged")
    fun setData(history : List<HistoryEntity>){
        this.historyList = history
        notifyDataSetChanged()
    }
}

