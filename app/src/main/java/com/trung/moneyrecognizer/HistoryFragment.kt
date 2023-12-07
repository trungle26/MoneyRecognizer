package com.trung.moneyrecognizer

import android.os.Bundle
import androidx.fragment.app.Fragment
import android.view.LayoutInflater
import android.view.View
import android.view.ViewGroup
import androidx.lifecycle.ViewModelProvider
import androidx.recyclerview.widget.LinearLayoutManager
import com.google.android.material.dialog.MaterialAlertDialogBuilder
import com.kimminh.moneysense.ui.history.HistoryAdapter
import com.kimminh.moneysense.ui.history.HistoryViewModel
import com.trung.moneyrecognizer.databinding.FragmentHistoryBinding

class HistoryFragment : Fragment() {
    private lateinit var  viewModel: HistoryViewModel

    private var _binding: FragmentHistoryBinding? = null

    // This property is only valid between onCreateView and
    // onDestroyView.
    private val binding get() = _binding!!

    override fun onCreateView(
        inflater: LayoutInflater,
        container: ViewGroup?,
        savedInstanceState: Bundle?
    ): View {
        _binding = FragmentHistoryBinding.inflate(inflater, container, false)
        val root: View = binding.root

        val historyAdapter= HistoryAdapter()
        val recyclerView = binding.rvHistory
        recyclerView.adapter = historyAdapter
        recyclerView.layoutManager = LinearLayoutManager(requireContext())

        viewModel = ViewModelProvider(this)[HistoryViewModel::class.java]
        viewModel.getAllHistory.observe(viewLifecycleOwner) { history ->
            historyAdapter.setData(history)
        }

        binding.btnDeleteAll.setOnClickListener{
            MaterialAlertDialogBuilder(requireContext())
                .setTitle(resources.getString(R.string.confirm_delete))
                .setMessage(resources.getString(R.string.delete_all_message))
                .setNegativeButton(resources.getString(R.string.cancel)) { dialog, _ ->
                    dialog.cancel()
                }
                .setPositiveButton(resources.getString(R.string.accept)) { _, _ ->
                    viewModel.deleteAll()
                }
                .show()
        }

        return root
    }


    override fun onDestroyView() {
        super.onDestroyView()
        _binding = null
    }
}